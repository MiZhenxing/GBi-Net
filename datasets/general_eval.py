from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, 
        ndepths=192, interval_scale=1.06, 
        img_mean=None, img_std=None, out_scale=1.0, self_norm=False, color_mode="RGB", 
        with_gt=False, max_h=None, max_w=None, base_image_size=None, use_cam_max=False,
        is_stage=False, stage_info=None, max_hw_mode="scale", out_scales=None, depth_num=4, **kwargs):        
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
        self.with_gt = with_gt
        self.max_h = max_h
        self.max_w = max_w
        self.base_image_size = base_image_size
        self.use_cam_max = use_cam_max 
        self.is_stage = is_stage
        self.stage_info = stage_info
        self.max_hw_mode = max_hw_mode
        self.out_scales = out_scales    
        self.depth_num = depth_num

        assert self.mode in ["test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) == 0:
                        continue
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
        depth_interval = float(lines[11].split()[1]) * self.interval_scale

        # if len(lines[11].split()) >= 3:
        if self.use_cam_max:
            depth_num = int(float(lines[11].split()[2]))
            depth_max = float(lines[11].split()[3])
            return intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max
        else:
            return intrinsics, extrinsics, depth_min, depth_interval
        
    def read_img_cv2(self, filename, color_mode=None):
        img = cv2.imread(filename)
        if color_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_img = np.array(img, dtype=np.float32)
        return np_img
    
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
        else:
            return intrinsics

    def getitem_stages(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        stage_num = len(self.stage_info["scale"])
        proj_matrices = {str(i):[] for i in range(stage_num)}
        cams = {str(i):[] for i in range(stage_num)}
        ref_imgs = {str(i):None for i in range(stage_num)}
        ref_cams = {str(i):None for i in range(stage_num)}
        depths = {str(i):None for i in range(stage_num)}
        masks = {str(i):None for i in range(stage_num)}

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            mask_filename = ""
            depth_filename = ""
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename, color_mode=self.color_mode)
            cam_data = self.read_cam_file(proj_mat_filename)
            if self.use_cam_max:
                intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max = cam_data
            elif len(cam_data) == 4:
                intrinsics, extrinsics, depth_min, depth_interval = cam_data
                
            if self.max_h is not None and self.max_w is not None:
                if self.max_hw_mode == "scale":
                    old_h = img.shape[0]
                    old_w = img.shape[1]
                    img = self.scale_img(img, self.max_h, self.max_w, interpolation=cv2.INTER_LINEAR)
                    intrinsics = self.scale_cam(intrinsics, h=old_h, w=old_w, max_h=self.max_h, max_w=self.max_w)
                elif self.max_hw_mode == "crop":
                    old_h = img.shape[0]
                    old_w = img.shape[1]
                    img = self.crop_img(img, new_h=self.max_h, new_w=self.max_w, base=self.base_image_size)
                    intrinsics = self.crop_cam(intrinsics, h=old_h, w=old_w, new_h=self.max_h, new_w=self.max_w, base=self.base_image_size)
            if self.base_image_size is not None:
                old_h = img.shape[0]
                old_w = img.shape[1]
                img = self.crop_img(img, new_h=None, new_w=None, base=self.base_image_size)
                intrinsics = self.crop_cam(intrinsics, h=old_h, w=old_w, new_h=None, new_w=None, base=self.base_image_size)
            if self.out_scale != 1.0:
                img = self.scale_img(img, scale=self.out_scale, interpolation=cv2.INTER_LINEAR)
                intrinsics = self.scale_cam(intrinsics, scale=self.out_scale)
            if self.out_scales is not None:
                tmp_h, tmp_w = img.shape[:2]
                tmp_h = tmp_h * self.out_scales[0] // self.out_scales[1]
                tmp_w = tmp_w * self.out_scales[0] // self.out_scales[1]
                img = self.scale_img(img, tmp_h, tmp_w, interpolation=cv2.INTER_LINEAR)
                intrinsics = self.scale_cam(intrinsics, scale=self.out_scales[0] / self.out_scales[1])
            
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
            
            img = self.norm_img(img, self_norm=self.self_norm, img_mean=self.img_mean, img_std=self.img_std)
            imgs.append(img)

            if i == 0:  # reference view
                depth_values = np.linspace(depth_min, depth_interval * self.ndepths + depth_min, num=self.ndepths, dtype=np.float32)
                # depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                #                          dtype=np.float32)
                if self.with_gt:
                    mask = self.read_img(mask_filename)
                    mask = self.norm_img(mask)   
                    mask[mask > 0.0] = 1.0
                    depth = self.read_depth(depth_filename)
                    
                    if self.max_h is not None and self.max_w is not None:
                        if self.max_hw_mode == "scale":
                            mask = self.scale_img(mask, self.max_h, self.max_w, interpolation=cv2.INTER_NEAREST)
                            depth = self.scale_img(depth, self.max_h, self.max_w, interpolation=cv2.INTER_NEAREST)
                        elif self.max_hw_mode == "crop":
                            mask = self.crop_img(mask, new_h=self.max_h, new_w=self.max_w, base=self.base_image_size)
                            depth = self.crop_img(depth, new_h=self.max_h, new_w=self.max_w, base=self.base_image_size)

                    if self.base_image_size is not None:
                        mask = self.crop_img(mask, new_h=None, new_w=None, base=self.base_image_size)
                        depth = self.crop_img(depth, new_h=None, new_w=None, base=self.base_image_size)
                    
                    if self.out_scale != 1.0:
                        depth = self.scale_img(img=depth, scale=self.out_scale, interpolation=cv2.INTER_NEAREST)
                        mask = self.scale_img(img=mask, scale=self.out_scale, interpolation=cv2.INTER_NEAREST)
                
                    # begin stages
                    for stage_id in range(stage_num):
                        stage_scale = self.stage_info["scale"][str(stage_id)]
                        stage_mask = self.scale_img(img=mask, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                        stage_depth = self.scale_img(img=depth, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                        masks[str(stage_id)] = stage_mask
                        depths[str(stage_id)] = stage_depth
                if self.use_cam_max:
                        depth_min_max = np.array([depth_min, depth_max], dtype=np.float32)
                else:
                    depth_min_max = np.array([depth_values[0], depth_values[-1]], dtype=np.float32)
                depth_range = depth_min_max[1] - depth_min_max[0]
                
        binary_tree =  np.zeros([2, ref_imgs[str(0)].shape[0], ref_imgs[str(0)].shape[1]], dtype=np.int64)
        binary_tree[0, :, :] = binary_tree[0, :, :] + 1
        # binary_tree[0] is level, binary_tree[1] is key
        depth_range = depth_min_max[1] - depth_min_max[0]
        sample_interval = depth_range / self.depth_num
        sample_depth = []
        for i in range(self.depth_num):
            sample_depth.append(np.ones([ref_imgs[str(0)].shape[0], ref_imgs[str(0)].shape[1]], dtype=np.float32) * (sample_interval * (i + i + 1) / 2.0 + depth_min))
            
        sample_depth = np.stack(sample_depth, axis=0)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = {str(j):np.stack(proj_matrices[str(j)], axis=0) for j in range(stage_num)}
        cams = {str(j):np.stack(cams[str(j)], axis=0) for j in range(stage_num)}

        if self.with_gt:
            return {"scan_name": scan,
                "img_id": ref_view,
                "ref_imgs": ref_imgs,
                "ref_cams": ref_cams,
                "imgs": imgs,
                "proj_matrices": proj_matrices,
                "cams": cams,
                "depths": depths,
                "depth_values": depth_values,
                "depth_min_max": depth_min_max,
                "binary_tree": {"tree": binary_tree, "depth": sample_depth},
                "masks": masks}
        else:
            return {"scan_name": scan,
                "img_id": ref_view,
                "ref_imgs": ref_imgs,
                "ref_cams": ref_cams,
                "imgs": imgs,
                "proj_matrices": proj_matrices,
                "cams": cams,
                "depth_values": depth_values,
                "depth_min_max": depth_min_max,
                "binary_tree": {"tree": binary_tree, "depth": sample_depth}
                }


    def __getitem__(self, idx):
        if self.is_stage:
            return self.getitem_stages(idx)
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        cams = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            mask_filename = ""
            depth_filename = ""
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img_cv2(img_filename, color_mode=self.color_mode)
            cam_data = self.read_cam_file(proj_mat_filename)
            if self.use_cam_max:
                intrinsics, extrinsics, depth_min, depth_interval, depth_num, depth_max = cam_data
            elif len(cam_data) == 4:
                intrinsics, extrinsics, depth_min, depth_interval = cam_data

            if self.max_h is not None and self.max_w is not None:
                old_h = img.shape[0]
                old_w = img.shape[1]
                img = self.scale_img(img, self.max_h, self.max_w, interpolation=cv2.INTER_LINEAR)
                intrinsics = self.scale_cam(intrinsics, h=old_h, w=old_w, max_h=self.max_h, max_w=self.max_w)
            if self.base_image_size is not None:
                old_h = img.shape[0]
                old_w = img.shape[1]
                img = self.crop_img(img, new_h=None, new_w=None, base=self.base_image_size)
                intrinsics = self.crop_cam(intrinsics, h=old_h, w=old_w, new_h=None, new_w=None, base=self.base_image_size)

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
                depth_values = np.linspace(depth_min, depth_interval * self.ndepths + depth_min, num=self.ndepths, dtype=np.float32)
                # depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                #                          dtype=np.float32)
                if self.with_gt:
                    mask = self.read_img(mask_filename)
                    mask = self.norm_img(mask)   
                    mask[mask > 0.0] = 1.0
                    depth = self.read_depth(depth_filename)
                    if self.max_h is not None and self.max_w is not None:
                        mask = self.scale_img(mask, self.max_h, self.max_w, interpolation=cv2.INTER_NEAREST)
                        depth = self.scale_img(depth, self.max_h, self.max_w, interpolation=cv2.INTER_NEAREST)
                    if self.base_image_size is not None:
                        mask = self.crop_img(mask, new_h=None, new_w=None, base=self.base_image_size)
                        depth = self.crop_img(depth, new_h=None, new_w=None, base=self.base_image_size)
                    
                    if self.out_scale != 1.0:
                        depth = self.scale_img(img=depth, scale=self.out_scale, interpolation=cv2.INTER_NEAREST)
                        mask = self.scale_img(img=mask, scale=self.out_scale, interpolation=cv2.INTER_NEAREST)
                if self.use_cam_max:
                    depth_min_max = np.array([depth_min, depth_max], dtype=np.float32)
                else:
                    depth_min_max = np.array([depth_values[0], depth_values[-1]], dtype=np.float32)

                depth_range = depth_min_max[1] - depth_min_max[0]
                binary_tree =  np.zeros([2, ref_img.shape[0], ref_img.shape[1]], dtype=np.int64)
                binary_tree[0, :, :] = binary_tree[0, :, :] + 1
                # binary_tree[0] is level, binary_tree[1] is key
            
                sample_interval = depth_range / 4.0
                sample_depth = []
                for i in range(4):
                    sample_depth.append(np.ones([ref_img.shape[0], ref_img.shape[1]], dtype=np.float32) * (sample_interval * (i + i + 1) / 2.0 + depth_min))
                
                sample_depth = np.stack(sample_depth, axis=0)
                                     
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        cams = np.stack(cams, axis=0)

        if self.with_gt:
            return {"scan_name": scan,
                    "img_id": ref_view,
                    "ref_img": ref_img,
                    "ref_cam": ref_cam,
                    "imgs": imgs,
                    "proj_matrices": proj_matrices,
                    "cams": cams,
                    "depth": depth,
                    "depth_values": depth_values,
                    "depth_min_max": depth_min_max,
                    "binary_tree": {"tree": binary_tree, "depth": sample_depth},
                    "mask": mask}
        
        else:
            return {"scan_name": scan,
                    "img_id": ref_view,
                    "ref_img": ref_img,
                    "ref_cam": ref_cam,
                    "imgs": imgs,
                    "proj_matrices": proj_matrices,
                    "cams": cams,
                    "depth_values": depth_values,
                    "depth_min_max": depth_min_max,
                    "binary_tree": {"tree": binary_tree, "depth": sample_depth},
                    }


