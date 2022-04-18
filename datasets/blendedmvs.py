from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import torch
from torchvision import transforms as T

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, 
        ndepths=192, interval_scale=1.06, 
        img_mean=None, img_std=None, out_scale=1.0, self_norm=False, color_mode="RGB", 
        is_stage=False, stage_info=None, random_view=False,
        random_crop=False, crop_h=576, crop_w=768, transform=True, depth_num=4, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.img_mean = img_mean
        self.img_std = img_std
        self.out_scale = out_scale
        self.self_norm = self_norm
        self.color_mode = color_mode
        self.is_stage = is_stage
        self.stage_info = stage_info
        self.random_view = random_view
        self.depth_num = depth_num

        if transform:
            self.transform = T.Compose([T.ColorJitter(brightness=0.25,
                                                contrast=0.5)
                                    ])
        else:
            self.transform = None
        self.random_crop = random_crop
        self.crop_h = crop_h
        self.crop_w = crop_w

        assert self.mode in ["train", "val"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            scan_path = os.path.join(self.datapath, scan)
            pair_file = "cams/pair.txt"
            # read the pair file
            with open(os.path.join(scan_path, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) >= self.nviews - 1:
                        metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        # depth_interval = float(lines[11].split()[1]) * self.interval_scale
        depth_interval = float(lines[11].split()[1])
        depth_num = int(float(lines[11].split()[2]))
        depth_max = float(lines[11].split()[3])
        return intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max

    def read_img(self, filename, color_mode=None):
        if color_mode == "BGR":
            img = cv2.imread(filename)
        elif color_mode == "RGB" or color_mode is None:
            img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        return np_img

    def norm_img(self, np_img, self_norm=False, img_mean=None, img_std=None):
        if self_norm:
            var = np.var(np_img, axis=(0, 1), keepdims=True)
            mean = np.mean(np_img, axis=(0, 1), keepdims=True)
            np_img = (np_img - mean) / (np.sqrt(var) + 1e-7)
            return np_img
        else:
            # scale 0~255 to 0~1
            np_img = np_img / 255.
            if (img_mean is not None) and (img_std is not None):
                # scale with given mean and std
                img_mean = np.array(img_mean, dtype=np.float32)
                img_std = np.array(img_std, dtype=np.float32)
                np_img = (np_img - img_mean) / img_std
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_img(self, img, max_h=None, max_w=None, scale=None, interpolation=cv2.INTER_LINEAR): 
        h, w = img.shape[:2]
        if scale:
            new_w, new_h = int(scale * w), int(scale * h)
            img = cv2.resize(img, [new_w, new_h], interpolation=interpolation)
        elif h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = int(scale * w), int(scale * h)
            img = cv2.resize(img, [new_w, new_h], interpolation=interpolation)
        return img
    
    def crop_img(self, img, new_h=None, new_w=None, base=8):
        h, w = img.shape[:2]

        if new_h is None or new_w is None:
            new_h = h // base * base
            new_w = w // base * base

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            img = img[start_h:finish_h, start_w:finish_w]
        return img
    
    def crop_img_any(self, img, start_h, start_w, new_h, new_w):
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        img = img[start_h:finish_h, start_w:finish_w]
        return img
    
    def crop_cam_any(self, intrinsics, start_h, start_w):
        new_intrinsics = np.copy(intrinsics)
        # principle point:
        new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
        new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
        return new_intrinsics
    
    def scale_cam(self, intrinsics, h=None, w=None, max_h=None, max_w=None, scale=None):
        if scale:
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0, :] *= scale
            new_intrinsics[1, :] *= scale
        elif h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0, :] *= scale
            new_intrinsics[1, :] *= scale
        return new_intrinsics
    
    def crop_cam(self, intrinsics, h, w, new_h=None, new_w=None, base=8):
        if new_h is None or new_w is None:
            new_h = h // base * base
            new_w = w // base * base

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
            new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
        return new_intrinsics
    
    def getitem_stages(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if not self.random_view:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
        else:
            num_src_views = len(src_views)
            rand_ids = torch.randperm(num_src_views)[:self.nviews - 1]
            src_views_t = torch.tensor(src_views)
            view_ids = [ref_view] + list(src_views_t[rand_ids].numpy())

        imgs = []
        stage_num = len(self.stage_info["scale"])
        proj_matrices = {str(i):[] for i in range(stage_num)}
        cams = {str(i):[] for i in range(stage_num)}
        ref_imgs = {str(i):None for i in range(stage_num)}
        ref_cams = {str(i):None for i in range(stage_num)}
        depths = {str(i):None for i in range(stage_num)}
        masks = {str(i):None for i in range(stage_num)}

        scan_path = os.path.join(self.datapath, scan)
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(scan_path,
                                        'blended_images/{:0>8}.jpg'.format(vid))
            depth_filename = os.path.join(scan_path, 'rendered_depth_maps/{:0>8}.pfm'.format(vid))
            proj_mat_filename = os.path.join(scan_path, 'cams/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename, color_mode=self.color_mode)
            
            intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max = self.read_cam_file(proj_mat_filename)
            
            # begin stages
            for stage_id in range(stage_num):
                stage_scale = self.stage_info["scale"][str(stage_id)]
                stage_intrinsics = self.scale_cam(intrinsics=intrinsics, scale=stage_scale)
                stage_proj_mat = extrinsics.copy()
                stage_proj_mat[:3, :4] = np.matmul(stage_intrinsics, stage_proj_mat[:3, :4])
                proj_matrices[str(stage_id)].append(stage_proj_mat)

                stage_cam = np.zeros([2, 4, 4], dtype=np.float32)
                stage_cam[0, :4, :4] = extrinsics
                stage_cam[1, :3, :3] = stage_intrinsics
                cams[str(stage_id)].append(stage_cam)
            
                if i == 0:  # reference view
                    stage_ref_img = img.copy()
                    stage_ref_img = self.scale_img(stage_ref_img, scale=stage_scale, interpolation=cv2.INTER_LINEAR)
                    stage_ref_img = np.array(stage_ref_img, dtype=np.uint8)
                    ref_imgs[str(stage_id)] = stage_ref_img
                    ref_cams[str(stage_id)] = stage_cam
            
            if self.mode == "train":
                if self.transform:
                    img = Image.fromarray(np.array(img, dtype=np.uint8))
                    img = self.transform(img)
                    img = np.array(img, dtype=np.float32)
            img = self.norm_img(img, self_norm=self.self_norm, img_mean=self.img_mean, img_std=self.img_std)
            imgs.append(img)

            if i == 0:  # reference view
                depth_values = np.linspace(depth_min, depth_max, num=depth_num, dtype=np.float32)
                gt_depth_interval = depth_values[1] - depth_values[0]
                depth = self.read_depth(depth_filename)
                mask = np.logical_and(depth > depth_min, depth < depth_max).astype(np.float32)
                depth_min_max = np.array([depth_min, depth_max], dtype=np.float32)

                depth_range = depth_min_max[1] - depth_min_max[0]
                # begin stages
                for stage_id in range(stage_num):
                    stage_scale = self.stage_info["scale"][str(stage_id)]
                    stage_mask = self.scale_img(img=mask, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                    stage_depth = self.scale_img(img=depth, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                    masks[str(stage_id)] = stage_mask
                    depths[str(stage_id)] = stage_depth

        binary_tree =  np.zeros([2, depths[str(0)].shape[0], depths[str(0)].shape[1]], dtype=np.int64)
        binary_tree[0, :, :] = binary_tree[0, :, :] + 1
            
        # binary_tree[0] is level, binary_tree[1] is key
        sample_interval = depth_range / self.depth_num
        sample_depth = []

        for i in range(self.depth_num):
            sample_depth.append(np.ones_like(depths[str(0)]) * (sample_interval * (i + i + 1) / 2.0 + depth_min))
        sample_depth = np.stack(sample_depth, axis=0)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = {str(j):np.stack(proj_matrices[str(j)], axis=0) for j in range(stage_num)}
        cams = {str(j):np.stack(cams[str(j)], axis=0) for j in range(stage_num)}

        return {"scan_name": scan,
                "img_id": ref_view,
                "ref_imgs": ref_imgs,
                "ref_cams": ref_cams,
                "imgs": imgs,
                "proj_matrices": proj_matrices,
                "cams": cams,
                "depths": depths,
                "depth_values": depth_values,
                "gt_depth_interval": gt_depth_interval,
                "depth_min_max": depth_min_max,
                "binary_tree": {"tree": binary_tree, "depth": sample_depth},
                "masks": masks}
    
    def getitem_stages_random_crop(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if not self.random_view:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
        else:
            num_src_views = len(src_views)
            rand_ids = torch.randperm(num_src_views)[:self.nviews - 1]
            src_views_t = torch.tensor(src_views)
            view_ids = [ref_view] + list(src_views_t[rand_ids].numpy())

        imgs = []
        stage_num = len(self.stage_info["scale"])
        proj_matrices = {str(i):[] for i in range(stage_num)}
        cams = {str(i):[] for i in range(stage_num)}
        ref_imgs = {str(i):None for i in range(stage_num)}
        ref_cams = {str(i):None for i in range(stage_num)}
        depths = {str(i):None for i in range(stage_num)}
        masks = {str(i):None for i in range(stage_num)}
        rand_start_h = None
        rand_start_w = None
        scan_path = os.path.join(self.datapath, scan)
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(scan_path,
                                        'blended_images/{:0>8}.jpg'.format(vid))
            depth_filename = os.path.join(scan_path, 'rendered_depth_maps/{:0>8}.pfm'.format(vid))
            proj_mat_filename = os.path.join(scan_path, 'cams/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename, color_mode=self.color_mode)
            intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max = self.read_cam_file(proj_mat_filename)

            if i == 0:  # reference view
                depth_values = np.linspace(depth_min, depth_max, num=depth_num, dtype=np.float32)
                gt_depth_interval = depth_values[1] - depth_values[0]
                depth = self.read_depth(depth_filename)
                mask = np.logical_and(depth > depth_min, depth < depth_max).astype(np.float32)
                depth_min_max = np.array([depth_min, depth_max], dtype=np.float32)
                depth_range = depth_min_max[1] - depth_min_max[0]

            h_o = img.shape[0]
            w_o = img.shape[1]
            if i == 0:  # reference view
                mask_o = mask.copy()
                if self.mode == "train":
                    while True:
                        rand_start_h = torch.randint(0, h_o - self.crop_h, [1]).item() # top
                        rand_start_w = torch.randint(0, w_o - self.crop_w, [1]).item() # left
                        tmp_mask = self.crop_img_any(img=mask_o, start_h=rand_start_h, start_w=rand_start_w, 
                            new_h=self.crop_h, new_w=self.crop_w)
                        # begin stages
                        for stage_id in range(stage_num):
                            stage_scale = self.stage_info["scale"][str(stage_id)]
                            stage_mask = self.scale_img(img=tmp_mask, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                            masks[str(stage_id)] = stage_mask
                        if np.any(masks[str(0)] > 0.0):
                            break
                if self.mode == "val":
                    rand_start_h = (h_o - self.crop_h) // 2
                    rand_start_w = (w_o - self.crop_w) // 2
                    tmp_mask = self.crop_img_any(img=mask_o, start_h=rand_start_h, start_w=rand_start_w, 
                        new_h=self.crop_h, new_w=self.crop_w)
                    # begin stages
                    for stage_id in range(stage_num):
                        stage_scale = self.stage_info["scale"][str(stage_id)]
                        stage_mask = self.scale_img(img=tmp_mask, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                        masks[str(stage_id)] = stage_mask

            img = self.crop_img_any(img=img, start_h=rand_start_h, start_w=rand_start_w, 
                new_h=self.crop_h, new_w=self.crop_w)
            intrinsics = self.crop_cam_any(intrinsics=intrinsics, start_h=rand_start_h, start_w=rand_start_w)
            if i == 0:  # reference view
                depth = self.crop_img_any(img=depth, start_h=rand_start_h, start_w=rand_start_w, 
                    new_h=self.crop_h, new_w=self.crop_w)

            # begin stages
            for stage_id in range(stage_num):
                stage_scale = self.stage_info["scale"][str(stage_id)]
                stage_intrinsics = self.scale_cam(intrinsics=intrinsics, scale=stage_scale)
                stage_proj_mat = extrinsics.copy()
                stage_proj_mat[:3, :4] = np.matmul(stage_intrinsics, stage_proj_mat[:3, :4])
                proj_matrices[str(stage_id)].append(stage_proj_mat)

                stage_cam = np.zeros([2, 4, 4], dtype=np.float32)
                stage_cam[0, :4, :4] = extrinsics
                stage_cam[1, :3, :3] = stage_intrinsics
                cams[str(stage_id)].append(stage_cam)
            
                if i == 0:  # reference view
                    stage_ref_img = img.copy()
                    stage_ref_img = self.scale_img(stage_ref_img, scale=stage_scale, interpolation=cv2.INTER_LINEAR)
                    stage_ref_img = np.array(stage_ref_img, dtype=np.uint8)
                    ref_imgs[str(stage_id)] = stage_ref_img
                    ref_cams[str(stage_id)] = stage_cam
                    stage_depth = self.scale_img(img=depth, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                    depths[str(stage_id)] = stage_depth
                    
            if self.mode == "train":
                if self.transform:
                    img = Image.fromarray(np.array(img, dtype=np.uint8))
                    img = self.transform(img)
                    img = np.array(img, dtype=np.float32)
            img = self.norm_img(img, self_norm=self.self_norm, img_mean=self.img_mean, img_std=self.img_std)
            imgs.append(img)

        binary_tree =  np.zeros([2, depths[str(0)].shape[0], depths[str(0)].shape[1]], dtype=np.int64)
        binary_tree[0, :, :] = binary_tree[0, :, :] + 1
        # binary_tree[0] is level, binary_tree[1] is key
                    
        depth_min = depth_min_max[0]
        sample_interval = depth_range / self.depth_num
        sample_depth = []
        for i in range(self.depth_num):
            sample_depth.append(np.ones_like(depths[str(0)]) * (sample_interval * (i + i + 1) / 2.0 + depth_min))
        
        sample_depth = np.stack(sample_depth, axis=0)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = {str(j):np.stack(proj_matrices[str(j)], axis=0) for j in range(stage_num)}
        cams = {str(j):np.stack(cams[str(j)], axis=0) for j in range(stage_num)}

        return {"scan_name": scan,
                "img_id": ref_view,
                "ref_imgs": ref_imgs,
                "ref_cams": ref_cams,
                "imgs": imgs,
                "proj_matrices": proj_matrices,
                "cams": cams,
                "depths": depths,
                "depth_values": depth_values,
                "gt_depth_interval": gt_depth_interval,
                "depth_min_max": depth_min_max,
                "binary_tree": {"tree": binary_tree, "depth": sample_depth},
                "masks": masks}

    def __getitem__(self, idx):
        if self.is_stage:
            if self.random_crop:
                return self.getitem_stages_random_crop(idx)
            else:
                return self.getitem_stages(idx)
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if not self.random_view:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
        else:
            num_src_views = len(src_views)
            rand_ids = torch.randperm(num_src_views)[:self.nviews - 1]
            src_views_t = torch.tensor(src_views)
            view_ids = [ref_view] + list(src_views_t[rand_ids].numpy())

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        cams = []

        scan_path = os.path.join(self.datapath, scan)
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(scan_path,
                                        'blended_images/{:0>8}.jpg'.format(vid))
            depth_filename = os.path.join(scan_path, 'rendered_depth_maps/{:0>8}.pfm'.format(vid))
            proj_mat_filename = os.path.join(scan_path, 'cams/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename, color_mode=self.color_mode)

            intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max = self.read_cam_file(proj_mat_filename)
            if self.out_scale != 1.0:
                intrinsics = self.scale_cam(intrinsics=intrinsics, scale=self.out_scale)
            
            if i == 0:  # reference view
                ref_img = img.copy()
                if self.out_scale != 1.0:
                    ref_img = self.scale_img(ref_img, scale=self.out_scale, interpolation=cv2.INTER_LINEAR)
                ref_img = np.array(ref_img, dtype=np.uint8)
                ref_cam = np.zeros(shape=(2, 4, 4), dtype=np.float32)
                ref_cam[0, :4, :4] = extrinsics
                ref_cam[1, :3, :3] = intrinsics
            
            if self.mode == "train":
                img = Image.fromarray(np.array(img, dtype=np.uint8))
                img = self.transform(img)
                img = np.array(img, dtype=np.float32)
            img = self.norm_img(img, self_norm=self.self_norm, img_mean=self.img_mean, img_std=self.img_std)
            imgs.append(img)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            cam = np.zeros([2, 4, 4], dtype=np.float32)
            cam[0, :4, :4] = extrinsics
            cam[1, :3, :3] = intrinsics
            cams.append(cam)

            if i == 0:  # reference view
                depth_values = np.linspace(depth_min, depth_max, num=depth_num, dtype=np.float32)
                gt_depth_interval = depth_values[1] - depth_values[0]
                depth = self.read_depth(depth_filename)
                if self.out_scale != 1.0:
                    depth = self.scale_img(img=depth, scale=self.out_scale, interpolation=cv2.INTER_NEAREST)
                mask = np.logical_and(depth > depth_min, depth < depth_max).astype(np.float32)
                depth_min_max = np.array([depth_min, depth_max], dtype=np.float32)

                depth_range = depth_min_max[1] - depth_min_max[0]
                binary_tree =  np.zeros([2, depth.shape[0], depth.shape[1]], dtype=np.int64)
                binary_tree[0, :, :] = binary_tree[0, :, :] + 1
                # binary_tree[0] is level, binary_tree[1] is key
                sample_interval = depth_range / 4.0
                sample_depth = []

                for i in range(4):
                    sample_depth.append(np.ones_like(depth) * (sample_interval * (i + i + 1) / 2.0 + depth_min))
                sample_depth = np.stack(sample_depth, axis=0)
                
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        cams = np.stack(cams, axis=0)

        return {"scan_name": scan,
                "img_id": ref_view,
                "ref_img": ref_img,
                "ref_cam": ref_cam,
                "imgs": imgs,
                "proj_matrices": proj_matrices,
                "cams": cams,
                "depth": depth,
                "depth_values": depth_values,
                "gt_depth_interval": gt_depth_interval,
                "depth_min_max": depth_min_max,
                "binary_tree": {"tree": binary_tree, "depth": sample_depth},
                "mask": mask}
