import argparse
import os
import os.path as osp
import logging
import time
import subprocess
import sys
import shutil

import random 
import numpy as np

import torch
import torch.nn as nn
import torch.optim

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import find_dataset_def
from models import find_model_def, find_loss_def

from utils.logger import setup_logger
from utils.config import load_config
from utils.functions import *
from datasets.data_io import read_pfm, save_pfm, write_cam
import cv2
import matplotlib.pyplot as plt
from utils.depthfusion import fusion_one
from utils.xy_fusion import filter_depth
from utils.collect_pointclouds import *

import pandas as pd

def test_model_stage(model,
                loss_fn,
                data_loader,
                max_tree_depth,
                depth2stage,
                out_depths,
                is_clean,
                save_depths,
                output_dir,
                my_rank,       
                logger,
                prob_depth=None,
                prob_out_depths=None,
                color_mode=None,
                log_period=1,
                tensorboard_logger=None
                ):
    
    if out_depths == "all":
        out_depths = list(range(1, max_tree_depth + 1))
    avg_test_scalars = {"depth{}".format(i): DictAverageMeter() for i in out_depths}
    all_test_scalars_dict = {"depth{}".format(i): [] for i in out_depths}
    
    model.eval()
    end = time.time()
    total_iteration = data_loader.__len__()
    max_stage_id = max(depth2stage.values())
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for iteration, sample in enumerate(data_loader):
            torch.cuda.reset_peak_memory_stats()
            data_time = time.time() - end
            end = time.time()
            global_step = iteration

            sample_cuda = tocuda(sample)
            pred_outs = model(sample_cuda, mode="all", max_tree_depth=max_tree_depth, is_clean=is_clean, out_depths=out_depths, prob_out_depths=prob_out_depths)
            forward_time = time.time() - end
            forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            
            prob_maps = pred_outs["pred_prob_dict"]
            prob_maps_prefix = {}
            last_prob = prob_maps["tree_depth_{}".format(max_tree_depth)]
            current_prob_prefix = 0.0
            for curr_tree_depth in range(1, max_tree_depth + 1):
                tmp_prob = prob_maps["tree_depth_{}".format(curr_tree_depth)]
                if tmp_prob.shape[1] != last_prob.shape[1] or tmp_prob.shape[2] != last_prob.shape[2]:
                    tmp_prob = F.interpolate(tmp_prob.unsqueeze(1), last_prob.shape[1:], mode="nearest").squeeze(1)
                current_prob_prefix = current_prob_prefix + tmp_prob
                prob_maps_prefix["tree_depth_{}".format(curr_tree_depth)] = current_prob_prefix
            prob_maps_prefix_mean = {}
            for curr_tree_depth in range(1, max_tree_depth + 1):
                prob_maps_prefix_mean["tree_depth_{}".format(curr_tree_depth)] = prob_maps_prefix["tree_depth_{}".format(curr_tree_depth)] / curr_tree_depth
                
            for curr_tree_depth in out_depths:
                stage_id = depth2stage[str(curr_tree_depth)]
                curr_depth_sample_num = 4
                next_depth_sample_num = 4
                scan_names = sample["scan_name"]
                img_ids = tensor2numpy(sample["img_id"])
                img_cams = tensor2numpy(sample["ref_cams"][str(stage_id)])
                ref_imgs = tensor2numpy(sample["ref_imgs"][str(stage_id)])
                
                tmp_curr_depth_imgs = pred_outs["current_depths_dict"]["tree_depth_{}".format(curr_tree_depth)]
                key_offset = next_depth_sample_num // 2 - 1
                pred_depth_maps = (tmp_curr_depth_imgs[:, key_offset] + tmp_curr_depth_imgs[:, key_offset + 1]) / 2.0
                # prob_maps = pred_outs["pred_prob_dict"]["tree_depth_{}".format(curr_tree_depth)]
                prob_maps = prob_maps_prefix_mean["tree_depth_{}".format(curr_tree_depth)]
                pred_depth_maps = tensor2numpy(pred_depth_maps)
                # prob_maps = tensor2numpy(prob_maps)

                avg_scalar_outputs = DictAverageMeter()
                batch_size = len(scan_names)
                for batch_idx in range(batch_size):
                    scan_name = scan_names[batch_idx]
                    img_id = img_ids[batch_idx]
                    scalar_outputs = None
                    reduced_scalar_outputs = None
                    if loss_fn is not None:
                        depth_gt = sample_cuda["depths"][str(stage_id)][batch_idx:batch_idx + 1]
                        mask = sample_cuda["masks"][str(stage_id)][batch_idx:batch_idx + 1]
                        gt_label = pred_outs["gt_label_dict"]["tree_depth_{}".format(curr_tree_depth)][batch_idx:batch_idx + 1]
                        gt_label = torch.squeeze(gt_label, 1)
                        pred_feature = pred_outs["pred_feature_dict"]["tree_depth_{}".format(curr_tree_depth)][batch_idx:batch_idx + 1]

                        loss_mask = pred_outs["mask_dict"]["tree_depth_{}".format(curr_tree_depth)][batch_idx:batch_idx + 1]
                        loss = loss_fn(pred_feature, gt_label, loss_mask)

                        pred_label = pred_outs["pred_label_dict"]["tree_depth_{}".format(curr_tree_depth)][batch_idx:batch_idx + 1]
                        tmp_curr_depth_imgs = pred_outs["current_depths_dict"]["tree_depth_{}".format(curr_tree_depth)][batch_idx:batch_idx + 1]
                        key_offset = next_depth_sample_num // 2 - 1

                        depth_est = (tmp_curr_depth_imgs[:, key_offset] + tmp_curr_depth_imgs[:, key_offset + 1]) / 2.0
                        # prob_map = pred_outs["pred_prob_dict"]["tree_depth_{}".format(curr_tree_depth)][batch_idx:batch_idx + 1]
                        prob_map = prob_maps[batch_idx:batch_idx + 1]

                        scalar_outputs = {"loss": loss}
                        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.0)

                        scalar_outputs["accu"] = GBiNet_accu(pred_label, gt_label, mask > 0.0)
                        for i in range(curr_depth_sample_num):
                            scalar_outputs["accu{}".format(i)] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, i), mask > 0.0))
                        
                        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 2)
                        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 4)
                        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 8)

                        scalar_outputs["thres2mm_accu"] = 1.0 - scalar_outputs["thres2mm_error"]
                        scalar_outputs["thres4mm_accu"] = 1.0 - scalar_outputs["thres4mm_error"]
                        scalar_outputs["thres8mm_accu"] = 1.0 - scalar_outputs["thres8mm_error"]
                        
                        scalar_outputs["front_prob"] = Prob_mean(prob_map, sample_cuda["masks"][str(depth2stage[str(max_tree_depth)])][batch_idx:batch_idx + 1] > 0.0)
                        scalar_outputs["back_prob"] = Prob_mean(prob_map, sample_cuda["masks"][str(depth2stage[str(max_tree_depth)])][batch_idx:batch_idx + 1] == 0.0)

                        scalar_outputs["front_back_diff"] = scalar_outputs["front_prob"] - scalar_outputs["back_prob"]
                        reduced_scalar_outputs = reduce_scalar_outputs(scalar_outputs)

                        scalar_outputs = tensor2float(scalar_outputs)
                        reduced_scalar_outputs = tensor2float(reduced_scalar_outputs)

                    if scalar_outputs:
                        logger.info(
                            " ".join(
                                [
                                    "Iter {}/{}".format(iteration, total_iteration),
                                    "scan_name {}".format(scan_name),
                                    "img_id {}".format(img_id),
                                    "Max_depth {}".format(max_tree_depth),
                                    "curr_depth {}".format(curr_tree_depth),
                                    "test loss {:.4f}".format(scalar_outputs["loss"]),
                                    "abs_depth_error {:.4f}".format(scalar_outputs["abs_depth_error"]),
                                    "thres2mm_error {:.4f}".format(scalar_outputs["thres2mm_error"]),
                                    "thres4mm_error {:.4f}".format(scalar_outputs["thres4mm_error"]),
                                    "thres8mm_error {:.4f}".format(scalar_outputs["thres8mm_error"]),
                                    "thres2mm_accu {:.4f}".format(scalar_outputs["thres2mm_accu"]),
                                    "thres4mm_accu {:.4f}".format(scalar_outputs["thres4mm_accu"]),
                                    "thres8mm_accu {:.4f}".format(scalar_outputs["thres8mm_accu"]),
                                    "front_prob {:.4f}".format(scalar_outputs["front_prob"]),
                                    "back_prob {:.4f}".format(scalar_outputs["back_prob"]),
                                    "front_back_diff {:.4f}".format(scalar_outputs["front_back_diff"]),
                                    "forward_time {:.4f}".format(forward_time),
                                    "data_time {:.4f}".format(data_time),
                                    "max_mem: {:.0f}".format(forward_max_memory_allocated)
                                ]
                            )
                        )
                    else:
                        logger.info(
                            " ".join(
                                [
                                    "Iter {}/{}".format(iteration, total_iteration),
                                    "scan_name {}".format(scan_name),
                                    "img_id {}".format(img_id),
                                    "forward_time {:.4f}".format(forward_time),
                                    "data_time {:.4f}".format(data_time),
                                    "max_mem: {:.0f}".format(forward_max_memory_allocated)
                                ]
                            )
                        )
                    test_scalars = {"forward_time": forward_time, "data_time": data_time, "max_mem": forward_max_memory_allocated}
                    if scalar_outputs:
                        test_scalars.update(scalar_outputs)
                        avg_scalar_outputs.update(reduced_scalar_outputs)
                    avg_test_scalars["depth{}".format(curr_tree_depth)].update(test_scalars)
                    test_scalars["scan_name"] = scan_name
                    test_scalars["img_id"] = img_id
                    all_test_scalars_dict["depth{}".format(curr_tree_depth)].append(test_scalars)
                
                if iteration % log_period == 0:
                    if tensorboard_logger is not None and my_rank == 0:
                        if scalar_outputs:
                            save_scalars(tensorboard_logger, 'test', avg_scalar_outputs.mean(), global_step)
            for curr_tree_depth in save_depths:
                stage_id = depth2stage[str(curr_tree_depth)]
                curr_depth_sample_num = 4
                next_depth_sample_num = 4
                depth_output_dir = osp.join(output_dir, "depth_{}".format(curr_tree_depth))
                os.makedirs(depth_output_dir, exist_ok=True)
                scan_names = sample["scan_name"]
                img_ids = tensor2numpy(sample["img_id"])
                img_cams = tensor2numpy(sample["ref_cams"][str(stage_id)])
                ref_imgs = tensor2numpy(sample["ref_imgs"][str(stage_id)])
                key_offset = next_depth_sample_num // 2 - 1
                tmp_curr_depth_imgs = pred_outs["current_depths_dict"]["tree_depth_{}".format(curr_tree_depth)]

                pred_depth_maps = (tmp_curr_depth_imgs[:, key_offset] + tmp_curr_depth_imgs[:, key_offset + 1]) / 2.0

                # prob_maps = pred_outs["pred_prob_dict"]["tree_depth_{}".format(curr_tree_depth)]
                prob_maps = prob_maps_prefix_mean["tree_depth_{}".format(curr_tree_depth)]
                pred_depth_maps = tensor2numpy(pred_depth_maps)
                prob_maps = tensor2numpy(prob_maps)

                for batch_idx in range(len(scan_names)):
                    scan_name = scan_names[batch_idx]
                    scan_folder = osp.join(depth_output_dir, scan_name)

                    if not osp.isdir(scan_folder):
                        os.makedirs(scan_folder, exist_ok=True)
                        logger.info("**** {} ****".format(scan_name))
                    
                    img_id = img_ids[batch_idx]
                    img_cam = img_cams[batch_idx]
                    depth_min_max = sample["depth_min_max"][batch_idx]

                    depth_start = depth_min_max[0]
                    depth_end = depth_min_max[1]

                    init_depth_map_path = osp.join(scan_folder, "{:0>8}_init.pfm".format(img_id))
                    out_ref_image_path = osp.join(scan_folder, "{:0>8}.jpg".format(img_id))

                    init_depth_map = pred_depth_maps[batch_idx]
                    if prob_depth is None:
                        prob_map = prob_maps[batch_idx]
                    else:
                        prob_map = prob_maps_prefix_mean["tree_depth_{}".format(prob_depth)][batch_idx].cpu().numpy()
                    ref_image = ref_imgs[batch_idx]

                    prob_map_path = osp.join(scan_folder, "{:0>8}_prob.pfm".format(img_id))
                    pred_prob_img_path = osp.join(scan_folder, "{:0>8}_prob_img.png".format(img_id))

                    save_pfm(prob_map_path, prob_map)        
                    plt.imsave(pred_prob_img_path, prob_map, vmin=0.0, vmax=1.0, cmap="rainbow")
                    save_pfm(init_depth_map_path, init_depth_map)
                    pred_depth_img_path = osp.join(scan_folder, "{:0>8}_pred.png".format(img_id))
                    plt.imsave(pred_depth_img_path, init_depth_map, cmap="rainbow", vmin=depth_start, vmax=depth_end)
                        
                    # plt.imsave(out_ref_image_path, ref_image)
                    if color_mode == "BGR":
                        cv2.imwrite(out_ref_image_path, ref_image)
                    else:
                        plt.imsave(out_ref_image_path, ref_image)
                    out_init_cam_path = osp.join(scan_folder, "cam_{:0>8}_init.txt".format(img_id))
                    write_cam(out_init_cam_path, img_cam)

                    if loss_fn is not None:
                        gt_depth_img_cpu = sample["depths"][str(stage_id)][batch_idx]
                        gt_depth_img_path = osp.join(scan_folder, "{:0>8}_gt.png".format(img_id))
                        plt.imsave(gt_depth_img_path, gt_depth_img_cpu, cmap="rainbow", vmin=depth_start, vmax=depth_end)    

            end = time.time()
        if tensorboard_logger is not None and my_rank == 0:
            for curr_tree_depth in out_depths:
                for key, value in avg_test_scalars["depth{}".format(curr_tree_depth)].mean().items():
                    tensorboard_logger.add_scalars("Avg_" + key + "_depth{}".format(curr_tree_depth), {"test": value}, global_step)
                logger.info("Depth {} average test scalars {}".format(curr_tree_depth, avg_test_scalars["depth{}".format(curr_tree_depth)].mean()))

    return avg_test_scalars, all_test_scalars_dict

def test_model_stage_profile(model,
                data_loader,
                max_tree_depth,
                depth2stage,
                out_depths,
                output_dir,
                logger,
                prob_depth=None,
                color_mode=None
                ):
    
    if out_depths == "all":
        out_depths = list(range(1, max_tree_depth + 1))
    avg_test_scalars = {"depth{}".format(i): DictAverageMeter() for i in out_depths}
    all_test_scalars_dict = {"depth{}".format(i): [] for i in out_depths}
    
    model.eval()
    end = time.time()
    total_iteration = data_loader.__len__()
    max_stage_id = max(depth2stage.values())
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for iteration, sample in enumerate(data_loader):
            torch.cuda.reset_peak_memory_stats()
            data_time = time.time() - end
            sample_cuda = tocuda(sample)
            end = time.time()
            pred_outs = model(sample_cuda, max_tree_depth=max_tree_depth, prob_out_depth=prob_depth)
            forward_time = time.time() - end
            del sample_cuda
            forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            
            curr_tree_depth = max_tree_depth
            scan_names = sample["scan_name"]
            img_ids = tensor2numpy(sample["img_id"])
            for batch_idx in range(len(scan_names)):
                scan_name = scan_names[batch_idx]
                img_id = img_ids[batch_idx]
                test_scalars = {"forward_time": forward_time, "data_time": data_time, "max_mem": forward_max_memory_allocated}

                avg_test_scalars["depth{}".format(curr_tree_depth)].update(test_scalars)
                test_scalars["scan_name"] = scan_name
                test_scalars["img_id"] = img_id
                all_test_scalars_dict["depth{}".format(curr_tree_depth)].append(test_scalars)

                logger.info(
                            " ".join(
                                [
                                    "Iter {}/{}".format(iteration, total_iteration),
                                    "scan_name {}".format(scan_name),
                                    "img_id {}".format(img_id),
                                    "forward_time {:.4f}".format(forward_time),
                                    "data_time {:.4f}".format(data_time),
                                    "max_mem: {:.0f}".format(forward_max_memory_allocated)
                                ]
                            )
                        )
            curr_tree_depth = max_tree_depth
            stage_id = depth2stage[str(curr_tree_depth)]
            depth_output_dir = osp.join(output_dir, "depth_{}".format(curr_tree_depth))
            os.makedirs(depth_output_dir, exist_ok=True)
            scan_names = sample["scan_name"]
            img_ids = tensor2numpy(sample["img_id"])
            img_cams = tensor2numpy(sample["ref_cams"][str(stage_id)])
            ref_imgs = tensor2numpy(sample["ref_imgs"][str(stage_id)])

            for batch_idx in range(len(scan_names)):
                scan_name = scan_names[batch_idx]
                scan_folder = osp.join(depth_output_dir, scan_name)

                if not osp.isdir(scan_folder):
                    os.makedirs(scan_folder, exist_ok=True)
                    logger.info("**** {} ****".format(scan_name))
                
                img_id = img_ids[batch_idx]
                img_cam = img_cams[batch_idx]
                depth_min_max = sample["depth_min_max"][batch_idx]

                depth_start = depth_min_max[0]
                depth_end = depth_min_max[1]

                init_depth_map = pred_outs[0][batch_idx].cpu().numpy()
                init_depth_map_path = osp.join(scan_folder, "{:0>8}_init.pfm".format(img_id))
                pred_depth_img_path = osp.join(scan_folder, "{:0>8}_pred.png".format(img_id))

                prob_map = pred_outs[1][batch_idx].cpu().numpy()
                prob_map_path = osp.join(scan_folder, "{:0>8}_prob.pfm".format(img_id))
                pred_prob_img_path = osp.join(scan_folder, "{:0>8}_prob_img.png".format(img_id))

                ref_image = ref_imgs[batch_idx]
                out_ref_image_path = osp.join(scan_folder, "{:0>8}.jpg".format(img_id))

                save_pfm(prob_map_path, prob_map)        
                plt.imsave(pred_prob_img_path, prob_map, vmin=0.0, vmax=1.0, cmap="rainbow")
                save_pfm(init_depth_map_path, init_depth_map)
                plt.imsave(pred_depth_img_path, init_depth_map, cmap="rainbow", vmin=depth_start, vmax=depth_end)
                    
                # plt.imsave(out_ref_image_path, ref_image)
                if color_mode == "BGR":
                    cv2.imwrite(out_ref_image_path, ref_image)
                else:
                    plt.imsave(out_ref_image_path, ref_image)
                out_init_cam_path = osp.join(scan_folder, "cam_{:0>8}_init.txt".format(img_id))
                write_cam(out_init_cam_path, img_cam)
        
            end = time.time()
    return avg_test_scalars, all_test_scalars_dict

def test(rank, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = cfg["master_port"]
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=cfg["world_size"])

    synchronize()
    set_random_seed(cfg["random_seed"])

    logger = setup_logger("gbinet_test{}".format(str(rank)), cfg["log_dir"], prefix="test")
    output_dir = cfg["output_dir"]
    torch.cuda.set_device(rank)

    state_dict = None

    if os.path.exists(cfg["model_path"]):
        loadckpt = os.path.join(cfg["model_path"])
        logger.info("Loading checkpoint from {}".format(loadckpt))
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
    else:
        logger.info("No checkpoint found in {}.".format(cfg["model_path"]))
        return

    MVSDataset = find_dataset_def(cfg["dataset"])

    model_def = find_model_def(cfg["model_file"], cfg["model_name"])
    model = model_def(cfg).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss_def = find_loss_def(cfg["model_file"], cfg.get("loss_name", cfg["model_name"] + "_loss"))
    
    if cfg["data"]["test"]["with_gt"]:
        model_loss = loss_def
    else:
        model_loss = None
    
    logger.info("Build model:\n{}".format(str(model)))
    model = DistributedDataParallel(model, device_ids=[rank])

    logger.info("Loading model ...")
    model.load_state_dict(state_dict['model'])

    if cfg.get("img_mean") and cfg.get("img_std"):
        img_mean = cfg.get("img_mean")
        img_std = cfg.get("img_std")
        logger.info("Mannual img_mean {} and img_std {}\n".format(img_mean, img_std))
    else:
        logger.info("No img_mean and img_std\n")
        img_mean = None
        img_std = None
    
    test_dataset = MVSDataset(cfg["data"]["test"]["root_dir"], 
        cfg["data"]["test"]["listfile"], "test", 
        cfg["data"]["test"]["num_view"], 
        cfg["data"]["test"]["num_depth"], 
        cfg["data"]["test"]["interval_scale"],
        img_mean=img_mean, img_std=img_std, 
        out_scale=cfg["data"]["test"]["out_scale"],
        self_norm=cfg["data"]["test"]["self_norm"],
        color_mode=cfg["data"]["test"]["color_mode"],
        with_gt=cfg["data"]["test"]["with_gt"], 
        max_h=cfg["data"]["test"]["max_h"], 
        max_w=cfg["data"]["test"]["max_w"], 
        base_image_size=cfg["data"]["test"]["base_image_size"],
        use_cam_max=cfg["data"]["test"].get("use_cam_max", False),
        is_stage=cfg["model"].get("is_stage", False), 
        stage_info=cfg["model"].get("stage_info", None),
        max_hw_mode=cfg["data"]["test"].get("max_hw_mode", "scale"),
        out_scales=cfg["data"]["test"].get("out_scales", None))

    test_sampler = DistributedSampler(test_dataset, num_replicas=cfg["world_size"], rank=rank, shuffle=False)
    test_data_loader = DataLoader(test_dataset, cfg["test"]["batch_size"], sampler=test_sampler, num_workers=cfg["data"]["num_workers"])

    if rank == 0:
        tensorboard_logger = SummaryWriter(cfg["log_dir"])
    else:
        tensorboard_logger = None
    
    # test
    logger.info("Start testing ...")
    
    if cfg.get("test_func_name", "test_model_stage") == "test_model_stage":
        avg_test_scalars, all_test_scalars_dict = test_model_stage(model,
            model_loss,
            data_loader=test_data_loader,
            max_tree_depth=cfg["max_depth"],
            depth2stage=cfg["model"]["stage_info"]["depth2stage"],
            depth2sample_num=cfg["model"]["stage_info"].get("depth2sample_num", None),
            output_dir=osp.join(cfg["output_dir"], "test_output"),
            is_clean=cfg["data"]["test"]["is_clean"],
            out_depths=cfg["data"]["test"]["out_depths"],
            save_depths=cfg["data"]["test"]["save_depths"],
            my_rank=rank,
            color_mode=cfg["data"]["test"]["color_mode"],
            logger=logging.getLogger("gbinet_test{}".format(str(rank)) + ".test"),
            log_period=cfg["test"]["log_period"],
            tensorboard_logger=tensorboard_logger
            )
    elif cfg.get("test_func_name", "test_model_stage") == "test_model_stage_profile":
        avg_test_scalars, all_test_scalars_dict = test_model_stage_profile(model,
                data_loader=test_data_loader,
                max_tree_depth=cfg["max_depth"],
                depth2stage=cfg["model"]["stage_info"]["depth2stage"],
                out_depths=cfg["data"]["test"]["out_depths"],
                output_dir=osp.join(cfg["output_dir"], "test_output"),
                logger=logging.getLogger("gbinet_test{}".format(str(rank)) + ".test"),
                prob_depth=cfg["data"]["test"].get("prob_depth", None),
                color_mode=cfg["data"]["test"]["color_mode"],
                )


    world_size = get_world_size()
    gathered_scalars = [None for i in range(world_size)]
    dist.all_gather_object(gathered_scalars, all_test_scalars_dict)

    if rank == 0:
        from itertools import chain
        out_depths = cfg["data"]["test"]["out_depths"]
        max_tree_depth = cfg["max_depth"]
        if out_depths == "all":
            out_depths = list(range(1, max_tree_depth + 1))
        for curr_tree_depth in out_depths:
            gathered_scalars_depth = [i["depth{}".format(curr_tree_depth)] for i in gathered_scalars]
            gathered_scalars_depth = list(chain.from_iterable(gathered_scalars_depth))
            test_df = pd.DataFrame(gathered_scalars_depth)
            test_df.to_csv(osp.join(cfg["log_dir"], "test_info_depth{}.csv".format(curr_tree_depth)), index=False)
            test_df_mean = test_df.mean(numeric_only=True).to_frame().T
            test_df_mean.to_csv(osp.join(cfg["log_dir"], "test_info_depth{}_mean.csv".format(curr_tree_depth)), index=False)
            print("depth {}".format(curr_tree_depth), test_df_mean)


def gipuma_filter(cfg):
    with open(cfg["data"]["test"]["listfile"]) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]
    data_folder = osp.join(cfg["output_dir"], "test_output")
    save_depths = cfg["data"]["test"]["save_depths"]
    target_suffixs = cfg["fusion"].get("target_suffix", "")
    for curr_tree_depth in save_depths:
        depth_data_folder = osp.join(data_folder, "depth_{}".format(curr_tree_depth))
        for para_id in range(len(cfg["fusion"]["prob_filter"])):
            prob_filter = cfg["fusion"]["prob_filter"][para_id]
            prob_threshold = cfg["fusion"]["prob_threshold"][para_id]
            disp_threshold = cfg["fusion"]["disp_threshold"][para_id]
            num_consistent = cfg["fusion"]["num_consistent"][para_id]
            if target_suffixs != "":
                target_suffix = target_suffixs[para_id]
            else:
                target_suffix = ""

            for scan in scans:
                dense_folder = os.path.join(depth_data_folder, scan)
                point_folder = fusion_one(dense_folder=dense_folder, fusibile_exe_path=cfg["fusion"]["fusibile_exe_path"], 
                    prob_filter=prob_filter, prob_threshold=prob_threshold, 
                    disp_threshold=disp_threshold, num_consistent=num_consistent)
            collect_args = type('tmp', (object,), {})()
            collect_args.target_dir = osp.join(cfg["output_dir"], "depth_{}".format(curr_tree_depth), "gipuma_filter"+target_suffix, "collected_points_{}".format(prob_threshold)) if prob_filter else osp.join(cfg["output_dir"], "depth_{}".format(curr_tree_depth), "gipuma_filter"+target_suffix, "collected_points")
            collect_args.root_dir = depth_data_folder
            collect_args.point_dir = point_folder.split("/")[-1]
            data_name = cfg.get("data_name", "dtu")
            if "dtu" in data_name:
                collect_dtu(collect_args)
            elif "tanks" in data_name:
                collect_args.tanks_log_dir = cfg["data"]["test"]["root_dir"]
                collect_tanks(collect_args)

def gipuma_filter_per(cfg):
    with open(cfg["data"]["test"]["listfile"]) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    data_folder = osp.join(cfg["output_dir"], "test_output")
    save_depths = cfg["data"]["test"]["save_depths"]
    for curr_tree_depth in save_depths:
        depth_data_folder = osp.join(data_folder, "depth_{}".format(curr_tree_depth))
        for para_id in range(cfg["fusion"]["gipuma_filter_per"]["para_num"]):
            for scan in scans:
                paras = cfg["fusion"]["gipuma_filter_per"][scan]
                prob_filter = paras["prob_filter"][para_id]
                prob_threshold = paras["prob_threshold"][para_id]
                disp_threshold = paras["disp_threshold"][para_id]
                num_consistent = paras["num_consistent"][para_id]
                dense_folder = os.path.join(depth_data_folder, scan)
                point_folder = fusion_one(dense_folder=dense_folder, fusibile_exe_path=cfg["fusion"]["gipuma_filter_per"]["fusibile_exe_path"], 
                    prob_filter=prob_filter, prob_threshold=prob_threshold, 
                    disp_threshold=disp_threshold, num_consistent=num_consistent)
            collect_args = type('tmp', (object,), {})()
            collect_args.target_dir = osp.join(cfg["output_dir"], "depth_{}".format(curr_tree_depth), "gipuma_filter_per", "collected_points_{}".format(prob_threshold)) \
                if prob_filter else osp.join(cfg["output_dir"], "depth_{}".format(curr_tree_depth), "gipuma_filter_per", "collected_points")
            collect_args.root_dir = depth_data_folder
            collect_args.point_dir = point_folder.split("/")[-1]
            data_name = cfg.get("data_name", "dtu")
            if "dtu" in data_name:
                collect_dtu(collect_args)
            elif "tanks" in data_name:
                collect_args.tanks_log_dir = cfg["data"]["test"]["root_dir"]
                collect_tanks(collect_args)

def xy_filter(rank, cfg):
    if cfg["fusion"]["xy_filter"].get("nprocs", None) is not None:
        scans = cfg["fusion"]["xy_filter"]["scans"][rank]
    else:
        with open(cfg["data"]["test"]["listfile"]) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

    data_folder = osp.join(cfg["output_dir"], "test_output")
    output_dir = cfg["fusion"]["xy_filter"].get("output_dir", cfg["output_dir"])
    save_depths = cfg["data"]["test"]["save_depths"]
    for curr_tree_depth in save_depths:
        depth_data_folder = osp.join(data_folder, "depth_{}".format(curr_tree_depth))
        for para_id in range(len(cfg["fusion"]["xy_filter"]["prob_threshold"])):
            prob_threshold = cfg["fusion"]["xy_filter"]["prob_threshold"][para_id]
            num_consistent = cfg["fusion"]["xy_filter"]["num_consistent"][para_id]
            img_dist_thresh = cfg["fusion"]["xy_filter"]["img_dist_thresh"][para_id] if cfg["fusion"]["xy_filter"].get("img_dist_thresh", None) is not None else 1.0
            depth_thresh = cfg["fusion"]["xy_filter"]["depth_thresh"][para_id] if cfg["fusion"]["xy_filter"].get("depth_thresh", None) is not None else 0.01

            point_dir = os.path.join(output_dir, str(para_id), "depth_{}".format(curr_tree_depth), "xy_filter", "collected_points_{}".format(prob_threshold)) \
                if prob_threshold != 0.0 else os.path.join(output_dir, str(para_id), "depth_{}".format(curr_tree_depth), "xy_filter", "collected_points")
            os.makedirs(point_dir, exist_ok=True)
            for scan in scans:
                scan_folder = os.path.join(depth_data_folder, scan)
                scan_data_folder = os.path.join(cfg["data"]["test"]["root_dir"], scan)
                if cfg["fusion"]["xy_filter"]["global_pair"]:
                    pair_path = osp.join(cfg["data"]["test"]["root_dir"], "Cameras/pair.txt")
                else:
                    pair_path = osp.join(scan_data_folder, "pair.txt")
                data_name = cfg.get("data_name", "dtu")
                if "dtu" in data_name:
                    scan_id = int(scan[4:])
                    ply_name = osp.join(point_dir, "binary_{:03d}_l3.ply".format(scan_id))
                elif "tanks" in data_name:
                    ply_name = osp.join(point_dir, "{}.ply".format(scan))
                filter_depth(scan_folder=scan_folder, pair_path=pair_path, plyfilename=ply_name,
                    prob_threshold=prob_threshold, num_consistent=num_consistent, img_dist_thresh=img_dist_thresh, depth_thresh=depth_thresh)

def xy_filter_per(rank, cfg):
    if cfg["fusion"]["xy_filter_per"].get("nprocs", None) is not None:
        scans = cfg["fusion"]["xy_filter_per"]["scans"][rank]
    else:
        with open(cfg["data"]["test"]["listfile"]) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

    data_folder = osp.join(cfg["output_dir"], "test_output")
    output_dir = cfg["fusion"]["xy_filter_per"].get("output_dir", cfg["output_dir"])
    save_depths = cfg["data"]["test"]["save_depths"]
    for curr_tree_depth in save_depths:
        depth_data_folder = osp.join(data_folder, "depth_{}".format(curr_tree_depth))
        for para_id in range(cfg["fusion"]["xy_filter_per"]["para_num"]):
            if cfg["fusion"]["xy_filter_per"].get("para_tag", None) is not None:
                para_tag = cfg["fusion"]["xy_filter_per"].get("para_tag")[para_id]
            else:
                para_tag = para_id
            for scan in scans:
                paras = cfg["fusion"]["xy_filter_per"][scan]
                prob_threshold = paras["prob_threshold"][para_tag]
                point_dir = os.path.join(output_dir, str(para_tag), "depth_{}".format(curr_tree_depth), "xy_filter_per", "collected_points_{}".format(prob_threshold)) \
                    if prob_threshold is not None else os.path.join(output_dir, str(para_tag), "depth_{}".format(curr_tree_depth), "xy_filter_per", "collected_points")
                num_consistent = paras["num_consistent"][para_tag]
                img_dist_thresh = paras["img_dist_thresh"][para_tag] if paras.get("img_dist_thresh", None) is not None else 1.0
                depth_thresh = paras["depth_thresh"][para_tag] if paras.get("depth_thresh", None) is not None else 0.01

                os.makedirs(point_dir, exist_ok=True)
                scan_folder = os.path.join(depth_data_folder, scan)
                scan_data_folder = os.path.join(cfg["data"]["test"]["root_dir"], scan)
                if cfg["fusion"]["xy_filter_per"]["global_pair"]:
                    pair_path = osp.join(cfg["data"]["test"]["root_dir"], "Cameras/pair.txt")
                else:
                    pair_path = osp.join(scan_data_folder, "pair.txt")
                data_name = cfg.get("data_name", "dtu")
                if "dtu" in data_name:
                    scan_id = int(scan[4:])
                    ply_name = osp.join(point_dir, "binary_{:03d}_l3.ply".format(scan_id))
                elif "tanks" in data_name:
                    ply_name = osp.join(point_dir, "{}.ply".format(scan))
                filter_depth(scan_folder=scan_folder, pair_path=pair_path, plyfilename=ply_name,
                    prob_threshold=prob_threshold, num_consistent=num_consistent, img_dist_thresh=img_dist_thresh, depth_thresh=depth_thresh)

def main():
    parser = argparse.ArgumentParser(description="PyTorch GBiNet Training")
    parser.add_argument("--cfg", dest="config_file", default="", metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["true_gpu"]
    output_dir = cfg["output_dir"]

    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    git_commit_id = \
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[0:-1]
    git_branch_name = \
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('UTF-8')[0:-1]

    num_gpus = len(cfg["gpu"])

    timestamp = time.strftime(".%m_%d_%H_%M_%S")
    log_dir = os.path.join(output_dir, "log{}".format(timestamp))

    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    cfg["config_file"] = args.config_file
    cfg["log_dir"] = log_dir

    # copy config file to log_dir
    shutil.copy(args.config_file, log_dir)

    logger = setup_logger("gbinet", log_dir, prefix="test")
    logger.info("Branch " + git_branch_name)
    logger.info("Commit ID " + git_commit_id)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(sys.argv))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    world_size = num_gpus
    cfg["world_size"] = world_size
    
    if not cfg.get("no_testing", False):
        mp.spawn(test,
            args=(cfg,),
            nprocs=world_size,
            join=True)

    if not cfg.get("no_fusion", False):
        if cfg["fusion"]["name"] == "gipuma_filter":
            logger.info("Gipuma filtering ...")
            gipuma_filter(cfg)
        elif cfg["fusion"]["name"] == "gipuma_filter_per":
            logger.info("Gipuma filtering ...")
            gipuma_filter_per(cfg)
        elif cfg["fusion"]["name"] == "xy_filter":
            logger.info("xy filtering ...")
            if cfg["fusion"]["xy_filter"].get("nprocs", None) is None:
                xy_filter(-1, cfg)
            else:
                with open(cfg["data"]["test"]["listfile"]) as f:
                    scans = f.readlines()
                    scans = [line.rstrip() for line in scans]
                cfg["fusion"]["xy_filter"]["scans"] = chunk_list(scans, cfg["fusion"]["xy_filter"]["nprocs"])
                mp.spawn(xy_filter,
                    args=(cfg,),
                    nprocs=cfg["fusion"]["xy_filter"]["nprocs"],
                    join=True)
        elif cfg["fusion"]["name"] == "xy_filter_per":
            logger.info("xy filtering ...")
            if cfg["fusion"]["xy_filter_per"].get("nprocs", None) is None:
                xy_filter_per(-1, cfg)
            else:
                with open(cfg["data"]["test"]["listfile"]) as f:
                    scans = f.readlines()
                    scans = [line.rstrip() for line in scans]
                cfg["fusion"]["xy_filter_per"]["scans"] = chunk_list(scans, cfg["fusion"]["xy_filter_per"]["nprocs"])
                mp.spawn(xy_filter_per,
                    args=(cfg,),
                    nprocs=cfg["fusion"]["xy_filter_per"]["nprocs"],
                    join=True)

    logger.info("All Done")


if __name__ == "__main__":
    main()
