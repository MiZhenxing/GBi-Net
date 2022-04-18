import torch
from utils import binary_tree
import torch.nn.functional as F

def update_4pred_4sample1(b_tree, pred_label, depth_start, depth_end, is_first, with_grad=False, no_detach=False):
    if not with_grad:
        with torch.no_grad():
            # 0 1 2 3
            # -1 0 1 2
            
            indicator = torch.ones_like(pred_label)
            direction = pred_label - 1
            if is_first:
                # indicator = torch.zeros_like(pred_label)
                direction = pred_label
            b_tree = binary_tree.update_tree1(b_tree, indicator, direction)
            
            if depth_start.dim() == 3:
                if depth_start.shape[1] != pred_label.shape[1] or pred_label.shape[2] != pred_label.shape[2]:
                    depth_start = torch.unsqueeze(depth_start, 1)
                    depth_start = F.interpolate(depth_start, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                    depth_start = torch.squeeze(depth_start, 1)

                    depth_end = torch.unsqueeze(depth_end, 1)
                    depth_end = F.interpolate(depth_end, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                    depth_end = torch.squeeze(depth_end, 1)
            elif depth_start.dim() == 1:
                depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
                depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
            
            depth_range = depth_end - depth_start
            next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 1.0))
            next_interval = depth_range / next_interval_num
            depthmap_list = []
            for i in range(4):
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

            curr_depth = torch.stack(depthmap_list, 1)
    else:
        # 0 1 2 3
        # -1 0 1 2
        
        indicator = torch.ones_like(pred_label)
        direction = pred_label - 1
        if is_first:
            # indicator = torch.zeros_like(pred_label)
            direction = pred_label
        b_tree = binary_tree.update_tree1(b_tree, indicator, direction)
        
        if depth_start.dim() == 3:
            if depth_start.shape[1] != pred_label.shape[1] or pred_label.shape[2] != pred_label.shape[2]:
                depth_start = torch.unsqueeze(depth_start, 1)
                depth_start = F.interpolate(depth_start, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                depth_start = torch.squeeze(depth_start, 1)

                depth_end = torch.unsqueeze(depth_end, 1)
                depth_end = F.interpolate(depth_end, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                depth_end = torch.squeeze(depth_end, 1)
        elif depth_start.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        
        depth_range = depth_end - depth_start
        next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 1.0))
        next_interval = depth_range / next_interval_num
        depthmap_list = []
        for i in range(4):
            tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
            tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
            tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

            depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

        curr_depth = torch.stack(depthmap_list, 1)
    if no_detach:
        return curr_depth, b_tree
    else:
        return curr_depth.detach(), b_tree.detach()


def get_four_label_l4_s4_bin(gt_depth_img, b_tree, depth_start, depth_end, is_first):
    with torch.no_grad():
        if depth_start.dim() == 1 or depth_end.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        bin_edge_list = []
        if is_first:
            bin_edge_list.append(torch.zeros_like(gt_depth_img) + depth_start)
            depth_range = depth_end - depth_start
            interval = depth_range / 4.0
            for i in range(4):
                bin_edge_list.append(bin_edge_list[0] + interval * (i + 1))
        else:
            depth_range = depth_end - depth_start
            
            next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 1))
            next_interval = depth_range / next_interval_num
            bin_edge_list = []
            for i in range(5):
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key = torch.minimum(tmp_key, next_interval_num + 1)
                bin_edge_list.append(next_interval * tmp_key + depth_start)
        
        gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device) - 1 
        for i in range(4):
            bin_mask = torch.ge(gt_depth_img, bin_edge_list[i])
            bin_mask = torch.logical_and(bin_mask, 
                torch.lt(gt_depth_img, bin_edge_list[i + 1]))
            gt_label[bin_mask] = i
        bin_mask = (gt_label != -1)
        return gt_label, bin_mask


def depthmap2tree(depth_img, tree_depth, depth_start, depth_end, scale_factor=1.0, mode='bilinear', with_grad=False, no_detach=False):
    if not with_grad:
        with torch.no_grad():
            if scale_factor != 1.0:
                depth_img = torch.unsqueeze(depth_img, 1)
                depth_img = F.interpolate(depth_img, scale_factor=scale_factor, mode=mode)
                depth_img = torch.squeeze(depth_img, 1)
            B, H, W = depth_img.shape
            b_tree =  torch.zeros([B, 2, H, W], \
                dtype=torch.int64, device=depth_img.device)
            b_tree[:, 0, :, :] = b_tree[:, 0, :, :] + tree_depth

            if depth_start.dim() == 3:
                if depth_start.shape[1] != depth_img.shape[1] or depth_start.shape[2] != depth_img.shape[2]:
                    depth_start = torch.unsqueeze(depth_start, 1)
                    depth_start = F.interpolate(depth_start, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                    depth_start = torch.squeeze(depth_start, 1)

                    depth_end = torch.unsqueeze(depth_end, 1)
                    depth_end = F.interpolate(depth_end, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                    depth_end = torch.squeeze(depth_end, 1)
            elif depth_start.dim() == 1:
                depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
                depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
            
            depth_range = depth_end - depth_start

            d_interval = depth_range / (2.0 ** tree_depth)
            b_tree[:, 1, :, :] = (torch.floor((depth_img - depth_start) / d_interval)).type(torch.int64)
            b_tree[:, 1, :, :] = torch.clamp(b_tree[:, 1, :, :], min=0, max=2 ** tree_depth)

            next_interval_num = torch.tensor(2.0 ** (tree_depth + 1.0), device=depth_img.device)
            next_interval = depth_range / next_interval_num
            depthmap_list = []

            for i in range(4):
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

            curr_depth = torch.stack(depthmap_list, 1)
    else:
        if scale_factor != 1.0:
            depth_img = torch.unsqueeze(depth_img, 1)
            depth_img = F.interpolate(depth_img, scale_factor=scale_factor, mode=mode)
            depth_img = torch.squeeze(depth_img, 1)
        B, H, W = depth_img.shape
        b_tree =  torch.zeros([B, 2, H, W], \
            dtype=torch.int64, device=depth_img.device)
        b_tree[:, 0, :, :] = b_tree[:, 0, :, :] + tree_depth

        if depth_start.dim() == 3:
            if depth_start.shape[1] != depth_img.shape[1] or depth_start.shape[2] != depth_img.shape[2]:
                depth_start = torch.unsqueeze(depth_start, 1)
                depth_start = F.interpolate(depth_start, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                depth_start = torch.squeeze(depth_start, 1)

                depth_end = torch.unsqueeze(depth_end, 1)
                depth_end = F.interpolate(depth_end, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                depth_end = torch.squeeze(depth_end, 1)
        elif depth_start.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        
        depth_range = depth_end - depth_start

        d_interval = depth_range / (2.0 ** tree_depth)
        b_tree[:, 1, :, :] = (torch.floor((depth_img - depth_start) / d_interval)).type(torch.int64)
        b_tree[:, 1, :, :] = torch.clamp(b_tree[:, 1, :, :], min=0, max=2 ** tree_depth)

        next_interval_num = torch.tensor(2.0 ** (tree_depth + 1.0), device=depth_img.device)
        next_interval = depth_range / next_interval_num
        depthmap_list = []

        for i in range(4):
            tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
            tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
            tmp_key1 = torch.minimum(tmp_key1, next_interval_num)

            depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

        curr_depth = torch.stack(depthmap_list, 1)
    if no_detach:
        return curr_depth, b_tree
    else:
        return curr_depth.detach(), b_tree.detach()
