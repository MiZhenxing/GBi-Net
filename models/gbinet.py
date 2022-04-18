import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.gbinet_submodules
from modules.gbinet_submodules import StageFeatExtNet, CostRegNet, CostRegNetBN, PixelwiseNet,FeatureFetcher, get_pixel_grids
import utils.depth_update as depth_update

class GBiNet(nn.Module):

    def __init__(self, cfg):
        super(GBiNet, self).__init__()
        self.cfg = cfg
        self.stage_num = cfg["model"]["stage_num"]
        self.output_channels = cfg["model"]["output_channels"]
        self.depth2stage = cfg["model"]["stage_info"]["depth2stage"]
        self.group_nums = cfg["model"]["group_nums"]
        self.feat_name = cfg["model"].get("feat_name", "StageFeatExtNet")
        self.feat_class = getattr(modules.gbinet_submodules, self.feat_name)
        self.img_feature = self.feat_class(base_channels=8, stage_num=self.stage_num, output_channels=self.output_channels)
        self.feature_fetcher = FeatureFetcher()
        
        if cfg["model"].get("use_3dbn", True):
            self.cost_network = nn.ModuleDict({
                str(i):CostRegNetBN(self.group_nums[i], 8) for i in range(self.stage_num)
            })
        else:
            self.cost_network = nn.ModuleDict({
                str(i):CostRegNet(self.group_nums[i], 8) for i in range(self.stage_num)
            })
        
        self.view_weight_nets = nn.ModuleDict({
                str(i):PixelwiseNet(self.group_nums[i]) for i in range(self.stage_num)
            })

    def sequential_wrapping(self, features, current_depths, feature_map_indices_grid, cam_intrinsic, cam_extrinsic, stage_id):
        ref_feature = features[0]
        num_views = len(features)
        B, C, H, W = ref_feature.shape
        depth_num = current_depths.shape[1]

        # ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, depth_num, 1, 1)
        # volume_sum = ref_volume
        # volume_sq_sum = ref_volume ** 2
        group_num = self.group_nums[stage_id]
        ref_feature = ref_feature.view(B, group_num, C//group_num, H, W)

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
        R = cam_extrinsic[:, :, :3, :3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW)
        
        cam_points = (uv * current_depths.view(B, depth_num, 1, -1))
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(B, 3, -1)  # (B, 3, D*FH*FW)

        num_world_points = world_points.size(-1)
        assert num_world_points == H * W * depth_num

        similarity_sum = 0.0
        pixel_wise_weight_sum = 0.0
        for src_idx in range(1, num_views):
            src_fea = torch.unsqueeze(features[src_idx], 1)
            src_cam_intrinsic = cam_intrinsic[:, src_idx:src_idx + 1]
            src_cam_extrinsic = cam_extrinsic[:, src_idx:src_idx + 1]
            warped_volume = self.feature_fetcher(src_fea, world_points, src_cam_intrinsic, src_cam_extrinsic)
            warped_volume = warped_volume.squeeze(1).view(B, C, depth_num, H, W)
            warped_volume = warped_volume.view(B, group_num, C//group_num, depth_num, H, W)
            similarity = (warped_volume * ref_feature.unsqueeze(3)).mean(2) # B, G, D, H, W
            view_weight = self.view_weight_nets[str(stage_id)](similarity) # B, 1, H, W

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, G, Ndepth, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) #[B,1,1,H,W]
            else:
                similarity_sum += similarity*view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume, similarity, view_weight
        
        similarity = similarity_sum.div_(pixel_wise_weight_sum)

        return similarity


    def forward_one_depth(self, data_batch, stage_id):

        img_list = data_batch["imgs"]
        cam_params_list = data_batch["cams"][str(stage_id)]

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()

        num_view = img_list.shape[1]

        img_feature_maps = []
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :]
            curr_feature_map = self.img_feature(curr_img)[str(stage_id)]
            img_feature_maps.append(curr_feature_map)
        
        current_depths = data_batch["binary_tree"]["depth"] # (B, D, H, W)

        B, C, H, W = img_feature_maps[0].shape        
        feature_map_indices_grid = get_pixel_grids(H, W) \
            .view(1, 1, 3, -1).expand(B, 1, 3, -1).to(img_feature_maps[0].device)

        cost_img = self.sequential_wrapping(img_feature_maps, current_depths, feature_map_indices_grid, cam_intrinsic=cam_intrinsic, cam_extrinsic=cam_extrinsic, stage_id=stage_id)

        pred_feature = self.cost_network[str(stage_id)](cost_img)
        pred_feature = torch.squeeze(pred_feature, 1)

        pred_label = torch.argmax(pred_feature, 1)
        pred_prob = torch.max(torch.softmax(pred_feature, 1), 1)[0]

        return {"pred_feature": pred_feature, "pred_label": pred_label, 
            "pred_prob": pred_prob}
    
    def forward_all_depth(self, data_batch, max_tree_depth, is_clean=True, out_depths=None, prob_out_depths=None):

        img_list = data_batch["imgs"]
        cam_params_list = data_batch["cams"]

        

        depth_min_max = data_batch["depth_min_max"]
        depth_start = depth_min_max[:, 0]
        depth_end = depth_min_max[:, 1]

        num_view = img_list.shape[1]

        img_feature_maps_dict = {str(i):[] for i in range(self.stage_num)}
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :]
            curr_feature_map = self.img_feature(curr_img)
            for j in range(self.stage_num):
                img_feature_maps_dict[str(j)].append(curr_feature_map[str(j)])

        is_first = True

        if not is_clean:
            pred_feature_dict = {}
            pred_label_dict = {}
            tree_dict = {}
            mask_dict = {}

        pred_prob_dict = {}
        current_depths_dict = {}
        gt_label_dict = {}
        
        current_depths = data_batch["binary_tree"]["depth"] # (B, D, H, W)
        current_tree = data_batch["binary_tree"]["tree"]
        bin_mask_prefix = torch.zeros([current_depths.shape[0], current_depths.shape[2], current_depths.shape[3]], device=current_depths.device) == 0
        for curr_tree_depth in range(1, max_tree_depth + 1):
            stage_id = self.depth2stage[str(curr_tree_depth)]
            img_feature_maps = img_feature_maps_dict[str(stage_id)]
            B, C, H, W = img_feature_maps[0].shape
            feature_map_indices_grid = get_pixel_grids(H, W) \
                .view(1, 1, 3, -1).expand(B, 1, 3, -1).to(img_list.device)
            cam_extrinsic = cam_params_list[str(stage_id)][:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
            cam_intrinsic = cam_params_list[str(stage_id)][:, :, 1, :3, :3].clone()

            cost_img = self.sequential_wrapping(img_feature_maps, current_depths, feature_map_indices_grid, cam_intrinsic=cam_intrinsic, cam_extrinsic=cam_extrinsic, stage_id=stage_id)

            pred_feature = self.cost_network[str(stage_id)](cost_img)
            pred_feature = torch.squeeze(pred_feature, 1)
            pred_label = torch.argmax(pred_feature, 1, keepdim=False)
            
            pred_prob = torch.max(torch.softmax(pred_feature, 1), 1, keepdim=False)

            if curr_tree_depth in prob_out_depths:
                pred_prob_dict["tree_depth_{}".format(curr_tree_depth)] = pred_prob[0]

            # before updation, depth maps and tree of the output of the last depth
            if curr_tree_depth in out_depths:
                if not "tree_depth_{}".format(curr_tree_depth - 1) in current_depths_dict:
                    current_depths_dict["tree_depth_{}".format(curr_tree_depth - 1)] = current_depths
            if not is_clean:  
                if curr_tree_depth in out_depths:
                    if not "tree_depth_{}".format(curr_tree_depth - 1) in tree_dict:
                        tree_dict["tree_depth_{}".format(curr_tree_depth - 1)] = current_tree

            if data_batch.get("depths", None) is not None:
                depth_gt = data_batch["depths"][str(stage_id)]
                mask = data_batch["masks"][str(stage_id)]
                gt_label, bin_mask = depth_update.get_four_label_l4_s4_bin(depth_gt, b_tree=current_tree, 
                    depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
                with torch.no_grad():
                    if (bin_mask_prefix.shape[1] != bin_mask.shape[1]) or (bin_mask_prefix.shape[2] != bin_mask.shape[2]):
                        bin_mask_prefix = torch.squeeze(F.interpolate(torch.unsqueeze(bin_mask_prefix.float(), 1), [bin_mask.shape[1], bin_mask.shape[2]], mode="nearest"), 1).bool()
                    bin_mask_prefix = torch.logical_and(bin_mask_prefix, bin_mask)
                    mask = torch.logical_and(bin_mask_prefix, mask > 0.0)
                    mask = mask.float()
                    if not is_clean:  
                        if curr_tree_depth in out_depths:
                            mask_dict["tree_depth_{}".format(curr_tree_depth)] = mask
            else:
                gt_label = None
            gt_label_dict["tree_depth_{}".format(curr_tree_depth)] = gt_label

            current_depths, current_tree = \
                depth_update.update_4pred_4sample1(current_tree, pred_label, depth_start, depth_end, is_first)
            is_first = False

            # depth maps after curr_tree_depth forward
            if curr_tree_depth in out_depths:
                current_depths_dict["tree_depth_{}".format(curr_tree_depth)] = current_depths

            if not is_clean:  
                if curr_tree_depth in out_depths:
                    pred_feature_dict["tree_depth_{}".format(curr_tree_depth)] = pred_feature
                    pred_label_dict["tree_depth_{}".format(curr_tree_depth)] = pred_label
                    tree_dict["tree_depth_{}".format(curr_tree_depth)] = current_tree
            depth_est = (current_depths[:, 1] + current_depths[:, 2]) / 2.0
            next_depth_stage = self.depth2stage[str(curr_tree_depth + 1)]
            if next_depth_stage != stage_id:
                current_depths, current_tree = \
                    depth_update.depthmap2tree(depth_est, curr_tree_depth + 1, depth_start=depth_start, 
                        depth_end=depth_end, scale_factor=2.0, mode='bilinear')

        if not is_clean:
            return {"pred_feature_dict":pred_feature_dict, "pred_label_dict":pred_label_dict, 
                "pred_prob_dict":pred_prob_dict, "gt_label_dict":gt_label_dict,
                "current_depths_dict":current_depths_dict, "tree_dict":tree_dict, "mask_dict":mask_dict}
        else:
            return {"pred_prob_dict":pred_prob_dict, "gt_label_dict":gt_label_dict,
                "current_depths_dict":current_depths_dict}

    def forward(self, data_batch, mode="one", max_tree_depth=None, stage_id=None, is_clean=True, out_depths=None, prob_out_depths=None, **kwargs):
        assert mode in ["one", "all"]
        if mode == "one":
            return self.forward_one_depth(data_batch, stage_id=stage_id)
        if mode == "all":
            if out_depths is None:
                out_depths = [max_tree_depth]
            if prob_out_depths is None:
                prob_out_depths = out_depths
            return self.forward_all_depth(data_batch, max_tree_depth, is_clean=is_clean, out_depths=out_depths, prob_out_depths=prob_out_depths)


def GBiNet_loss(preds, gt_label, mask):
    # preds: B, C, H, W
    # gt_label: B, H, W
    # mask: B, H, W
    mask = mask > 0.0 # B, H, W
    preds = preds.permute(0, 2, 3, 1) # B, H, W, C
    preds_mask = preds[mask] # N, C
    gt_label_mask = gt_label[mask] # N
    loss = F.cross_entropy(preds_mask, gt_label_mask, reduction='mean')
    return loss