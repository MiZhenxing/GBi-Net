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
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer

from utils.logger import setup_logger
from utils.config import load_config
from utils.functions import *
import utils.depth_update as depth_update


def train_model_stage(model,
                loss_fn,
                data_loader,
                optimizer,
                scheduler,
                max_tree_depth,
                depth2stage,
                curr_epoch,
                my_rank,                 
                logger,
                log_period=1,
                tensorboard_logger=None
                ):
    avg_train_scalars = {"depth{}".format(i): DictAverageMeter() for i in range(1, max_tree_depth + 1)}
    torch.cuda.reset_peak_memory_stats()
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()

    for iteration, sample in enumerate(data_loader):
        data_time = time.time() - end
        end = time.time()
        global_step = total_iteration * curr_epoch + iteration
        sample_cuda = tocuda(sample)
        
        bin_mask_prefix = torch.zeros_like(sample_cuda["masks"][str(0)]) == 0
        is_first = True
        for curr_tree_depth in range(1, max_tree_depth + 1):
            stage_id = depth2stage[str(curr_tree_depth)]
            optimizer.zero_grad()
            
            outputs = model(data_batch=sample_cuda, mode="one", stage_id=stage_id)
                
            depth_gt = sample_cuda["depths"][str(stage_id)]
            mask = sample_cuda["masks"][str(stage_id)]
            
            depth_min_max = sample_cuda["depth_min_max"]
            gt_label, bin_mask = depth_update.get_four_label_l4_s4_bin(depth_gt, b_tree=sample_cuda["binary_tree"]["tree"], 
                depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
            # gt_label = torch.squeeze(gt_label, 1)
            with torch.no_grad():
                if (bin_mask_prefix.shape[1] != bin_mask.shape[1]) or (bin_mask_prefix.shape[2] != bin_mask.shape[2]):
                    bin_mask_prefix = torch.squeeze(F.interpolate(torch.unsqueeze(bin_mask_prefix.float(), 1), [bin_mask.shape[1], bin_mask.shape[2]], mode="nearest"), 1).bool()
                bin_mask_prefix = torch.logical_and(bin_mask_prefix, bin_mask)
                mask = torch.logical_and(bin_mask_prefix, mask > 0.0).float()

            preds = outputs["pred_feature"]
            loss = loss_fn(preds, gt_label, mask)
            loss.backward()
            optimizer.step()

            pred_label = outputs["pred_label"]

            depth_min_max = sample_cuda["depth_min_max"]
            sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                depth_update.update_4pred_4sample1(sample_cuda["binary_tree"]["tree"], 
                torch.unsqueeze(pred_label, 1), depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
            
            depth_est = (sample_cuda["binary_tree"]["depth"][:, 1] + sample_cuda["binary_tree"]["depth"][:, 2]) / 2.0

            is_first = False
            next_depth_stage = depth2stage[str(curr_tree_depth + 1)]
            if next_depth_stage != stage_id:
                depth_min_max = sample_cuda["depth_min_max"]
                sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                    depth_update.depthmap2tree(depth_est, curr_tree_depth + 1, depth_start=depth_min_max[:, 0], 
                        depth_end=depth_min_max[:, 1], scale_factor=2.0, mode='bilinear')

            prob_map = outputs["pred_prob"]

            scalar_outputs = {"loss": loss}

            depth_est_mapped = mapping_color(depth_est, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")
            depth_gt_mapped = mapping_color(depth_gt, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")

            image_outputs = {"depth_est": depth_est_mapped, "depth_gt": depth_gt_mapped,
                            "ref_img": sample["ref_imgs"][str(stage_id)].permute(0, 3, 1, 2) / 255.,
                            "mask": mask.cpu()}
            image_outputs["ori_mask"] = sample["masks"][str(stage_id)]
                            
            image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
            scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.0)

            scalar_outputs["accu"] = GBiNet_accu(pred_label, gt_label, mask > 0.0)
            scalar_outputs["accu0"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 0), mask > 0.0))
            scalar_outputs["accu1"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 1), mask > 0.0))
            scalar_outputs["accu2"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 2), mask > 0.0))
            scalar_outputs["accu3"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 3), mask > 0.0))
            
            scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 2)
            scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 4)
            scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 8)
            
            scalar_outputs["thres2mm_accu"] = 1.0 - scalar_outputs["thres2mm_error"]
            scalar_outputs["thres4mm_accu"] = 1.0 - scalar_outputs["thres4mm_error"]
            scalar_outputs["thres8mm_accu"] = 1.0 - scalar_outputs["thres8mm_error"]

            scalar_outputs["front_prob"] = Prob_mean(prob_map, mask > 0.0)
            scalar_outputs["back_prob"] = Prob_mean(prob_map, mask == 0.0)

            scalar_outputs = reduce_scalar_outputs(scalar_outputs)

            loss = tensor2float(loss)
            scalar_outputs = tensor2float(scalar_outputs)

            forward_time = time.time() - end
            avg_train_scalars["depth{}".format(curr_tree_depth)].update(scalar_outputs)

            if iteration % log_period == 0:
                mode = 'train'
                if tensorboard_logger is not None and my_rank == 0:
                    for key, value in scalar_outputs.items():
                        name = '{}/{}_depth{}'.format(mode, key, curr_tree_depth)
                        tensorboard_logger.add_scalar(name, value, global_step)
                
                logger.info(
                " ".join(
                        [
                            "Epoch {}".format(curr_epoch),
                            "Iter {}/{}".format(iteration, total_iteration),
                            "Max_depth {}".format(max_tree_depth),
                            "curr_depth {}".format(curr_tree_depth),
                            "train loss {:.4f}".format(loss),

                            "accu {:.4f}".format(scalar_outputs["accu"]),
                            "accu0 {:.4f}".format(scalar_outputs["accu0"]),
                            "accu1 {:.4f}".format(scalar_outputs["accu1"]),
                            "accu2 {:.4f}".format(scalar_outputs["accu2"]),
                            "accu3 {:.4f}".format(scalar_outputs["accu3"]),

                            "abs_depth_error {:.4f}".format(scalar_outputs["abs_depth_error"]),
                            "thres2mm_error {:.4f}".format(scalar_outputs["thres2mm_error"]),
                            "thres4mm_error {:.4f}".format(scalar_outputs["thres4mm_error"]),
                            "thres8mm_error {:.4f}".format(scalar_outputs["thres8mm_error"]),
                            "thres2mm_accu {:.4f}".format(scalar_outputs["thres2mm_accu"]),
                            "thres4mm_accu {:.4f}".format(scalar_outputs["thres4mm_accu"]),
                            "thres8mm_accu {:.4f}".format(scalar_outputs["thres8mm_accu"]),
                            "front_prob {:.4f}".format(scalar_outputs["front_prob"]),
                            "back_prob {:.4f}".format(scalar_outputs["back_prob"]),
                            "forward_time {:.4f}".format(forward_time),
                            "data_time {:.4f}".format(data_time),
                            "lr {}".format([a["lr"] for a in optimizer.param_groups]),
                            "max_mem: {:.0f}".format(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                        ]
                    )
                )

            end = time.time()

        if tensorboard_logger is not None and my_rank == 0 and iteration % (log_period * 5) == 0:
            save_images(tensorboard_logger, 'train', image_outputs, global_step)
        if scheduler is not None:
            scheduler.step_update(curr_epoch * total_iteration + iteration)
        end = time.time()

    if tensorboard_logger is not None and my_rank == 0:
        for curr_tree_depth in range(1, max_tree_depth + 1):
            for key, value in avg_train_scalars["depth{}".format(curr_tree_depth)].mean().items():
                tensorboard_logger.add_scalars("Avg_" + key + "_depth{}".format(curr_tree_depth), {"train": value}, curr_epoch)
            logger.info("Epoch {} depth {} average train scalars {}".format(curr_epoch, curr_tree_depth, avg_train_scalars["depth{}".format(curr_tree_depth)].mean()))


def train_model_stage_blended(model,
                loss_fn,
                data_loader,
                optimizer,
                scheduler,
                max_tree_depth,
                depth2stage,
                curr_epoch,
                my_rank,                 
                logger,
                log_period=1,
                tensorboard_logger=None
                ):
    avg_train_scalars = {"depth{}".format(i): DictAverageMeter() for i in range(1, max_tree_depth + 1)}
    torch.cuda.reset_peak_memory_stats()
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()

    for iteration, sample in enumerate(data_loader):
        data_time = time.time() - end
        end = time.time()
        global_step = total_iteration * curr_epoch + iteration
        sample_cuda = tocuda(sample)

        is_first = True
        bin_mask_prefix = torch.zeros_like(sample_cuda["masks"][str(0)]) == 0
        for curr_tree_depth in range(1, max_tree_depth + 1):
            stage_id = depth2stage[str(curr_tree_depth)]
            optimizer.zero_grad()

            outputs = model(sample_cuda, mode="one", stage_id=stage_id)
            
            depth_gt = sample_cuda["depths"][str(stage_id)]
            mask = sample_cuda["masks"][str(stage_id)]
            
            depth_min_max = sample_cuda["depth_min_max"]
            gt_label, bin_mask = depth_update.get_four_label_l4_s4_bin(depth_gt, b_tree=sample_cuda["binary_tree"]["tree"], 
                depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
            # gt_label = torch.squeeze(gt_label, 1)
            with torch.no_grad():
                if (bin_mask_prefix.shape[1] != bin_mask.shape[1]) or (bin_mask_prefix.shape[2] != bin_mask.shape[2]):
                    bin_mask_prefix = torch.squeeze(F.interpolate(torch.unsqueeze(bin_mask_prefix.float(), 1), [bin_mask.shape[1], bin_mask.shape[2]], mode="nearest"), 1).bool()
                bin_mask_prefix = torch.logical_and(bin_mask_prefix, bin_mask)
                mask = torch.logical_and(bin_mask_prefix, mask > 0.0).float()
            
            preds = outputs["pred_feature"]

            if torch.any(mask > 0.0):
                loss = loss_fn(preds, gt_label, mask)
                loss.backward()
                optimizer.step()
            else:
                # skip optimizer.step() if mask has no valid values
                # https://discuss.pytorch.org/t/how-to-skip-backward-if-loss-is-very-small-in-training/52759
                mask[:, :, 0] = 1.0
                loss = loss_fn(preds, gt_label, mask)
                loss.backward()
                optimizer.zero_grad()

            pred_label = outputs["pred_label"]

            depth_min_max = sample_cuda["depth_min_max"]
            sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                depth_update.update_4pred_4sample1(sample_cuda["binary_tree"]["tree"], 
                torch.unsqueeze(pred_label, 1), depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)

            depth_est = (sample_cuda["binary_tree"]["depth"][:, 1] + sample_cuda["binary_tree"]["depth"][:, 2]) / 2.0

            is_first = False
            next_depth_stage = depth2stage[str(curr_tree_depth + 1)]
            if next_depth_stage != stage_id:
                depth_min_max = sample_cuda["depth_min_max"]
                sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                    depth_update.depthmap2tree(depth_est, curr_tree_depth + 1, depth_start=depth_min_max[:, 0], 
                        depth_end=depth_min_max[:, 1], scale_factor=2.0, mode='bilinear')

            gt_depth_interval = sample_cuda["gt_depth_interval"]
            prob_map = outputs["pred_prob"]

            scalar_outputs = {"loss": loss}

            depth_est_mapped = mapping_color(depth_est, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")
            depth_gt_mapped = mapping_color(depth_gt, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")
            
            image_outputs = {"depth_est": depth_est_mapped, "depth_gt": depth_gt_mapped,
                            "ref_img": sample["ref_imgs"][str(stage_id)].permute(0, 3, 1, 2) / 255.,
                            "mask": mask.cpu()}

            image_outputs["ori_mask"] = sample["masks"][str(stage_id)]

            image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
            scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.0)

            scalar_outputs["accu"] = GBiNet_accu(pred_label, gt_label, mask > 0.0)
            scalar_outputs["accu0"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 0), mask > 0.0))
            scalar_outputs["accu1"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 1), mask > 0.0))
            scalar_outputs["accu2"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 2), mask > 0.0))
            scalar_outputs["accu3"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 3), mask > 0.0))

            scalar_outputs["thres1_error"] = Batch_Thres_metrics(depth_est, depth_gt, mask > 0.0, gt_depth_interval)
            scalar_outputs["thres2_error"] = Batch_Thres_metrics(depth_est, depth_gt, mask > 0.0, 2 * gt_depth_interval)
            scalar_outputs["thres4_error"] = Batch_Thres_metrics(depth_est, depth_gt, mask > 0.0, 4 * gt_depth_interval)
            
            scalar_outputs["thres1_accu"] = 1.0 - scalar_outputs["thres1_error"]
            scalar_outputs["thres2_accu"] = 1.0 - scalar_outputs["thres2_error"]
            scalar_outputs["thres4_accu"] = 1.0 - scalar_outputs["thres4_error"]

            # scalar_outputs["front_prob"] = Prob_mean(prob_map, mask > 0.0)
            # scalar_outputs["back_prob"] = Prob_mean(prob_map, mask == 0.0)

            scalar_outputs = reduce_scalar_outputs(scalar_outputs)

            loss = tensor2float(loss)
            scalar_outputs = tensor2float(scalar_outputs)

            forward_time = time.time() - end
            avg_train_scalars["depth{}".format(curr_tree_depth)].update(scalar_outputs)

            if iteration % log_period == 0:
                mode = 'train'
                if tensorboard_logger is not None and my_rank == 0:
                    for key, value in scalar_outputs.items():
                        name = '{}/{}_depth{}'.format(mode, key, curr_tree_depth)
                        tensorboard_logger.add_scalar(name, value, global_step)
                
                logger.info(
                " ".join(
                        [
                            "Epoch {}".format(curr_epoch),
                            "Iter {}/{}".format(iteration, total_iteration),
                            "Max_depth {}".format(max_tree_depth),
                            "curr_depth {}".format(curr_tree_depth),
                            "train loss {:.4f}".format(loss),

                            "accu {:.4f}".format(scalar_outputs["accu"]),
                            "accu0 {:.4f}".format(scalar_outputs["accu0"]),
                            "accu1 {:.4f}".format(scalar_outputs["accu1"]),
                            "accu2 {:.4f}".format(scalar_outputs["accu2"]),
                            "accu3 {:.4f}".format(scalar_outputs["accu3"]),

                            "abs_depth_error {:.4f}".format(scalar_outputs["abs_depth_error"]),
                            "thres1_error {:.4f}".format(scalar_outputs["thres1_error"]),
                            "thres2_error {:.4f}".format(scalar_outputs["thres2_error"]),
                            "thres4_error {:.4f}".format(scalar_outputs["thres4_error"]),
                            "thres1_accu {:.4f}".format(scalar_outputs["thres1_accu"]),
                            "thres2_accu {:.4f}".format(scalar_outputs["thres2_accu"]),
                            "thres4_accu {:.4f}".format(scalar_outputs["thres4_accu"]),
                            # "front_prob {:.4f}".format(scalar_outputs["front_prob"]),
                            # "back_prob {:.4f}".format(scalar_outputs["back_prob"]),
                            "forward_time {:.4f}".format(forward_time),
                            "data_time {:.4f}".format(data_time),
                            "lr {}".format([a["lr"] for a in optimizer.param_groups]),
                            "max_mem: {:.0f}".format(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                        ]
                    )
                )

            end = time.time()

        if tensorboard_logger is not None and my_rank == 0 and iteration % (log_period * 5) == 0:
            save_images(tensorboard_logger, 'train', image_outputs, global_step)
        if scheduler is not None:
            scheduler.step_update(curr_epoch * total_iteration + iteration)
        end = time.time()

    if tensorboard_logger is not None and my_rank == 0:
        for curr_tree_depth in range(1, max_tree_depth + 1):
            for key, value in avg_train_scalars["depth{}".format(curr_tree_depth)].mean().items():
                tensorboard_logger.add_scalars("Avg_" + key + "_depth{}".format(curr_tree_depth), {"train": value}, curr_epoch)
            logger.info("Epoch {} depth {} average train scalars {}".format(curr_epoch, curr_tree_depth, avg_train_scalars["depth{}".format(curr_tree_depth)].mean()))


def validate_model_stage(model,
                    loss_fn,
                    data_loader,
                    max_tree_depth,
                    depth2stage,
                    curr_epoch, 
                    my_rank,       
                    logger,
                    log_period=1,
                    tensorboard_logger=None
                   ):
    avg_test_scalars = {"depth{}".format(i): DictAverageMeter() for i in range(1, max_tree_depth + 1)}
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    end = time.time()
    total_iteration = data_loader.__len__()
    with torch.no_grad():
        for iteration, sample in enumerate(data_loader):
            data_time = time.time() - end
            end = time.time()
            global_step = total_iteration * curr_epoch + iteration
            sample_cuda = tocuda(sample)

            bin_mask_prefix = torch.zeros_like(sample_cuda["masks"][str(0)]) == 0
            is_first = True
            for curr_tree_depth in range(1, max_tree_depth + 1):
                stage_id = depth2stage[str(curr_tree_depth)]
                
                outputs = model(data_batch=sample_cuda, mode="one", stage_id=stage_id, depth_id=curr_tree_depth)
                
                if "view_weight_list" in outputs:
                    sample_cuda["view_weight_list"] = outputs["view_weight_list"]
                
                depth_gt = sample_cuda["depths"][str(stage_id)]
                mask = sample_cuda["masks"][str(stage_id)]

                depth_min_max = sample_cuda["depth_min_max"]
                gt_label, bin_mask = depth_update.get_four_label_l4_s4_bin(depth_gt, b_tree=sample_cuda["binary_tree"]["tree"], 
                    depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
                # gt_label = torch.squeeze(gt_label, 1)
                with torch.no_grad():
                    if (bin_mask_prefix.shape[1] != bin_mask.shape[1]) or (bin_mask_prefix.shape[2] != bin_mask.shape[2]):
                        bin_mask_prefix = torch.squeeze(F.interpolate(torch.unsqueeze(bin_mask_prefix.float(), 1), [bin_mask.shape[1], bin_mask.shape[2]], mode="nearest"), 1).bool()
                    bin_mask_prefix = torch.logical_and(bin_mask_prefix, bin_mask)
                    mask = torch.logical_and(bin_mask_prefix, mask > 0.0).float()
                
                preds = outputs["pred_feature"]
                loss = loss_fn(preds, gt_label, mask)

                pred_label = outputs["pred_label"]
            
                depth_min_max = sample_cuda["depth_min_max"]
                sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                    depth_update.update_4pred_4sample1(sample_cuda["binary_tree"]["tree"], 
                    torch.unsqueeze(pred_label, 1), depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
                
                depth_est = (sample_cuda["binary_tree"]["depth"][:, 1] + sample_cuda["binary_tree"]["depth"][:, 2]) / 2.0

                is_first = False
                next_depth_stage = depth2stage[str(curr_tree_depth + 1)]
                if next_depth_stage != stage_id:
                    depth_min_max = sample_cuda["depth_min_max"]
                    sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                        depth_update.depthmap2tree(depth_est, curr_tree_depth + 1, depth_start=depth_min_max[:, 0], 
                            depth_end=depth_min_max[:, 1], scale_factor=2.0, mode='bilinear')
                
                prob_map = outputs["pred_prob"]

                scalar_outputs = {"loss": loss}
                
                depth_est_mapped = mapping_color(depth_est, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")
                depth_gt_mapped = mapping_color(depth_gt, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")
                
                image_outputs = {"depth_est": depth_est_mapped, "depth_gt": depth_gt_mapped,
                                "ref_img": sample["ref_imgs"][str(stage_id)].permute(0, 3, 1, 2) / 255.,
                                "mask": mask.cpu()}
                
                image_outputs["ori_mask"] = sample["masks"][str(stage_id)]
                
                image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
                scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.0)

                scalar_outputs["accu"] = GBiNet_accu(pred_label, gt_label, mask > 0.0)
                scalar_outputs["accu0"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 0), mask > 0.0))
                scalar_outputs["accu1"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 1), mask > 0.0))
                scalar_outputs["accu2"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 2), mask > 0.0))
                scalar_outputs["accu3"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 3), mask > 0.0))
                
                scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 2)
                scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 4)
                scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 8)

                scalar_outputs["thres2mm_accu"] = 1.0 - scalar_outputs["thres2mm_error"]
                scalar_outputs["thres4mm_accu"] = 1.0 - scalar_outputs["thres4mm_error"]
                scalar_outputs["thres8mm_accu"] = 1.0 - scalar_outputs["thres8mm_error"]

                scalar_outputs["front_prob"] = Prob_mean(prob_map, mask > 0.0)
                scalar_outputs["back_prob"] = Prob_mean(prob_map, mask == 0.0)
                
                scalar_outputs = reduce_scalar_outputs(scalar_outputs)

                loss = tensor2float(loss)
                scalar_outputs = tensor2float(scalar_outputs)

                forward_time = time.time() - end
                avg_test_scalars["depth{}".format(curr_tree_depth)].update(scalar_outputs)

                if iteration % log_period == 0:
                    mode = 'val'
                    if tensorboard_logger is not None and my_rank == 0:
                        for key, value in scalar_outputs.items():
                            name = '{}/{}_depth{}'.format(mode, key, curr_tree_depth)
                            tensorboard_logger.add_scalar(name, value, global_step)
                    
                    logger.info(
                    " ".join(
                            [
                                "Epoch {}".format(curr_epoch),
                                "Iter {}/{}".format(iteration, total_iteration),
                                "Max_depth {}".format(max_tree_depth),
                                "curr_depth {}".format(curr_tree_depth),
                                "val loss {:.4f}".format(loss),

                                "accu {:.4f}".format(scalar_outputs["accu"]),
                                "accu0 {:.4f}".format(scalar_outputs["accu0"]),
                                "accu1 {:.4f}".format(scalar_outputs["accu1"]),
                                "accu2 {:.4f}".format(scalar_outputs["accu2"]),
                                "accu3 {:.4f}".format(scalar_outputs["accu3"]),

                                "abs_depth_error {:.4f}".format(scalar_outputs["abs_depth_error"]),
                                "thres2mm_error {:.4f}".format(scalar_outputs["thres2mm_error"]),
                                "thres4mm_error {:.4f}".format(scalar_outputs["thres4mm_error"]),
                                "thres8mm_error {:.4f}".format(scalar_outputs["thres8mm_error"]),
                                "thres2mm_accu {:.4f}".format(scalar_outputs["thres2mm_accu"]),
                                "thres4mm_accu {:.4f}".format(scalar_outputs["thres4mm_accu"]),
                                "thres8mm_accu {:.4f}".format(scalar_outputs["thres8mm_accu"]),
                                "front_prob {:.4f}".format(scalar_outputs["front_prob"]),
                                "back_prob {:.4f}".format(scalar_outputs["back_prob"]),
                                "forward_time {:.4f}".format(forward_time),
                                "data_time {:.4f}".format(data_time),
                                "max_mem: {:.0f}".format(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                            ]
                        )
                    )

                end = time.time()

            if tensorboard_logger is not None and my_rank == 0 and iteration % (log_period * 5) == 0:
                save_images(tensorboard_logger, 'val', image_outputs, global_step)

        if tensorboard_logger is not None and my_rank == 0:
            for curr_tree_depth in range(1, max_tree_depth + 1):
                for key, value in avg_test_scalars["depth{}".format(curr_tree_depth)].mean().items():
                    tensorboard_logger.add_scalars("Avg_" + key + "_depth{}".format(curr_tree_depth), {"val": value}, curr_epoch)
                logger.info("Epoch {} depth {} average val scalars {}".format(curr_epoch, curr_tree_depth, avg_test_scalars["depth{}".format(curr_tree_depth)].mean()))

    return avg_test_scalars["depth{}".format(max_tree_depth)].mean()


def train(rank, cfg):
    if cfg.get("slurm", False):
        slurm_global_rank = int(os.environ['SLURM_PROCID'])
        # rank = 0
        slurm_local_rank = int(os.environ['SLURM_LOCALID'])
        logger = setup_logger("gbinet_train{}".format(str(slurm_local_rank)), cfg["log_dir"], prefix="train")
        world_size = cfg["world_size"]
        iplist = os.environ['SLURM_JOB_NODELIST']
        print("iplist", iplist)
        print("device_count", torch.cuda.device_count())
        def get_ip(iplist):
            if not '[' in iplist:
                return iplist
            ip = iplist.split('[')[0] + iplist.split('[')[1].split('-')[0]
            return ip
        slurm_ips = get_ip(iplist)
        # logger.info(slurm_ips, slurm_global_rank, slurm_local_rank, world_size)
        def dist_init(host_addr, rank, local_rank, world_size, port=23456):
            host_addr_full = 'tcp://' + host_addr + ':' + str(port)
            torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                                rank=rank, world_size=world_size)
            assert torch.distributed.is_initialized()
        dist_init(slurm_ips, slurm_global_rank, slurm_local_rank, world_size, port=cfg["master_port"])
        torch.cuda.set_device(slurm_local_rank)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = cfg["master_port"]
        world_size = cfg["world_size"]
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        synchronize()
        logger = setup_logger("gbinet_train{}".format(str(rank)), cfg["log_dir"], prefix="train")
        torch.cuda.set_device(rank)
        
    output_dir = cfg["output_dir"]
    set_random_seed(cfg["random_seed"])
    state_dict = None
    if cfg["auto_resume"] or cfg.get("model_path"): 
        if cfg.get("model_path"):
            if os.path.exists(cfg["model_path"]):
                loadckpt = os.path.join(cfg["model_path"])
                logger.info("Loading checkpoint from {}".format(loadckpt))
                state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
            else:
                logger.info("No checkpoint found in {}. Initializing model from scratch".format(cfg["model_path"]))
        elif cfg["auto_resume"]:
            saved_models = [fn for fn in os.listdir(output_dir) if (fn.endswith(".ckpt") and (not fn.endswith("best.ckpt")))]
            if saved_models:
                saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                # use the latest checkpoint file
                loadckpt = os.path.join(output_dir, saved_models[-1])
                logger.info("Loading checkpoint from {}".format(loadckpt))
                state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
            else:
                logger.info("No checkpoint found for auto_resume. Initializing model from scratch")
    else:
        logger.info("No checkpoint found. Initializing model from scratch")

    
    model_def = find_model_def(cfg["model_file"], cfg["model_name"])
    if cfg.get("slurm", False):
        model = model_def(cfg).to("cuda")
    else:
        model = model_def(cfg).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss_def = find_loss_def(cfg["model_file"], cfg.get("loss_name", cfg["model_name"] + "_loss"))
    model_loss = loss_def
    
    logger.info("Build model:\n{}".format(str(model)))
    if cfg.get("slurm", False):
        model = DistributedDataParallel(model, device_ids=[slurm_local_rank],
            find_unused_parameters=cfg.get("find_unused_parameters", False), broadcast_buffers=cfg.get("broadcast_buffers", True))
    else:
        model = DistributedDataParallel(model, device_ids=[rank], 
            find_unused_parameters=cfg.get("find_unused_parameters", False), broadcast_buffers=cfg.get("broadcast_buffers", True))

    optimizer = build_optimizer(cfg, model)

    if cfg.get("img_mean") and cfg.get("img_std"):
        img_mean = cfg.get("img_mean")
        img_std = cfg.get("img_std")
        logger.info("Mannual img_mean {} and img_std {}\n".format(img_mean, img_std))
    else:
        logger.info("No img_mean and img_std\n")
        img_mean = None
        img_std = None
    
    MVSDataset = find_dataset_def(cfg["dataset"])
    
    train_dataset = MVSDataset(cfg["data"]["train"]["root_dir"], 
        cfg["data"]["train"]["listfile"], "train", 
        cfg["data"]["train"]["num_view"], 
        cfg["data"]["train"]["num_depth"], 
        cfg["data"]["train"]["interval_scale"],
        img_mean=img_mean, img_std=img_std,
        out_scale=cfg["data"]["out_scale"],
        self_norm=cfg["data"]["train"]["self_norm"],
        color_mode=cfg["data"]["train"]["color_mode"],
        is_stage=cfg["model"].get("is_stage", False), 
        stage_info=cfg["model"].get("stage_info", None),
        random_view=cfg["data"]["train"].get("random_view", False),
        img_interp=cfg["data"]["train"].get("img_interp", "linear"),
        random_crop=cfg["data"]["train"].get("random_crop", False),
        crop_h=cfg["data"]["train"].get("crop_h", 512),
        crop_w=cfg["data"]["train"].get("crop_w", 640),
        depth_num=cfg["data"]["train"].get("depth_num", 4),
        transform=cfg["data"]["train"].get("transform", True))
    if cfg["data"]["val"].get("dataset", None) is not None:
        ValMVSDataset = find_dataset_def(cfg["data"]["val"].get("dataset"))
    else:
        ValMVSDataset = MVSDataset
    val_dataset = ValMVSDataset(cfg["data"]["val"]["root_dir"], 
        cfg["data"]["val"]["listfile"], "val", 
        cfg["data"]["val"]["num_view"], 
        cfg["data"]["val"]["num_depth"], 
        cfg["data"]["val"]["interval_scale"],
        img_mean=img_mean, img_std=img_std,
        out_scale=cfg["data"]["out_scale"],
        self_norm=cfg["data"]["val"]["self_norm"],
        color_mode=cfg["data"]["val"]["color_mode"],
        is_stage=cfg["model"].get("is_stage", False), 
        stage_info=cfg["model"].get("stage_info", None),
        img_interp=cfg["data"]["val"].get("img_interp", "linear"),
        random_crop=cfg["data"]["val"].get("random_crop", False),
        crop_h=cfg["data"]["val"].get("crop_h", 512),
        crop_w=cfg["data"]["val"].get("crop_w", 640),
        depth_num=cfg["data"]["val"].get("depth_num", 4))

    if cfg.get("slurm", False):
        rank = slurm_global_rank
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_data_loader = DataLoader(train_dataset, cfg["train"]["batch_size"], sampler=train_sampler, num_workers=cfg["data"]["num_workers"])
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_data_loader = DataLoader(val_dataset, cfg["val"]["batch_size"], sampler=val_sampler, num_workers=cfg["data"]["num_workers"])

    if cfg.get("slurm", False):
        # rank = slurm_local_rank
        rank = dist.get_rank()
        # https://github.com/facebookresearch/maskrcnn-benchmark/pull/40
        # https://github.com/pytorch/pytorch/issues/12042

    scheduler = build_scheduler(cfg, optimizer, len(train_data_loader))

    if state_dict:
        logger.info("Loading model ...")
        model.load_state_dict(state_dict['model'])
        if cfg["model"].get("finetune", False):
            start_epoch = 0
            best_metric_name = cfg["train"]["val_metric"]
            best_metric = None
        else:
            logger.info("Loading optimizer ...")
            optimizer.load_state_dict(state_dict['optimizer'])
            if scheduler is not None:
                logger.info("Loading scheduler ...")
                scheduler.load_state_dict(state_dict['scheduler'])
            start_epoch = state_dict['epoch'] + 1 # state_dict['epoch'] is the epoch id of the model, from 0
            best_metric_name = cfg["train"]["val_metric"]
            best_metric = state_dict[best_metric_name]
    else:
        start_epoch = 0
        best_metric_name = cfg["train"]["val_metric"]
        best_metric = None
    
    if cfg["scheduler"].get("change"):
        # now only for multi-step scheduler
        from collections import Counter
        logger.info("Changing scheduler ...")
        scheduler.milestones=Counter(cfg["scheduler"]["milestones"])
        scheduler.gamma=cfg["scheduler"]["gamma"]
    
    if rank == 0:
        tensorboard_logger = SummaryWriter(cfg["log_dir"])
    else:
        tensorboard_logger = None
    
    # train    
    max_epoch = cfg["train"]["max_epoch"]
    ckpt_period = cfg["train"]["checkpoint_period"]

    logger.info("Start training from epoch {}".format(start_epoch))
    max_max_tree_depth = cfg["max_depth"]
    for epoch in range(start_epoch, max_epoch):
        train_sampler.set_epoch(epoch)

        init_max_depth = cfg["model"].get("init_max_depth", 2)
        max_tree_depth = min(epoch + init_max_depth, max_max_tree_depth)
        if "blended" in cfg["dataset"]:
            train_func = train_model_stage_blended
        else:
            train_func = train_model_stage
        train_func(model,
            model_loss,
            data_loader=train_data_loader,
            optimizer=optimizer,
            scheduler=None if cfg["scheduler"]["name"] == 'multi_step' else scheduler,
            max_tree_depth=max_tree_depth,
            depth2stage=cfg["model"]["stage_info"]["depth2stage"],
            curr_epoch=epoch,
            my_rank=rank,
            logger=logging.getLogger("gbinet_train{}".format(str(rank)) + ".train"),
            log_period=cfg["train"]["log_period"],
            tensorboard_logger=tensorboard_logger
            )

        if cfg["scheduler"]["name"] == 'multi_step':
            scheduler.step()
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

        # checkpoint
        if rank == 0:
            if epoch % ckpt_period == 0 or (epoch + 1) == max_epoch:
                logger=logging.getLogger("gbinet_train{}".format(str(rank)) + ".train")
                logger.info("Saving trained model of epoch {} ...".format(epoch))
                if scheduler is not None:
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        best_metric_name: best_metric
                        },
                        osp.join(output_dir, "model_{:03d}.ckpt".format(epoch)))
                else:
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        best_metric_name: best_metric
                        },
                        osp.join(output_dir, "model_{:03d}.ckpt".format(epoch)))
        
        val_period = cfg["train"]["val_period"]
        if epoch % val_period == 0 or (epoch + 1) == max_epoch:
            avg_test_scalar = validate_model_stage(model,
                model_loss,
                data_loader=val_data_loader,
                max_tree_depth=max_tree_depth,
                depth2stage=cfg["model"]["stage_info"]["depth2stage"],
                curr_epoch=epoch,
                my_rank=rank,           
                logger=logging.getLogger("gbinet_train{}".format(str(rank)) + ".val"),
                log_period=cfg["val"]["log_period"],
                tensorboard_logger=tensorboard_logger
                )

            # best validation
            cur_metric = avg_test_scalar[best_metric_name]
            if best_metric is None or cur_metric > best_metric:
                best_metric = cur_metric
                if rank == 0:
                    logger=logging.getLogger("gbinet_train{}".format(str(rank)) + ".train")
                    logger.info("Saving best model of epoch {} ...".format(epoch))
                    if scheduler is not None:
                        torch.save({
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            best_metric_name: best_metric
                        },
                        osp.join(output_dir, "model_best.ckpt")) 
                    else:
                        torch.save({
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            best_metric_name: best_metric
                        },
                        osp.join(output_dir, "model_best.ckpt")) 

            logger.info("Best val-{} = {}".format(best_metric_name, best_metric))


def main():
    parser = argparse.ArgumentParser(description="PyTorch GBiNet Training")
    parser.add_argument("--cfg", dest="config_file", default="", metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    if cfg.get("slurm", False):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0" # workaround of no gpu bug
        output_dir = cfg["output_dir"]
        slurm_USERDIR = os.environ.get('TACC_USERDIR')
        cfg["output_dir"] = osp.join(slurm_USERDIR, output_dir)        
        output_dir = cfg["output_dir"]
        cfg["data"]["train"]["root_dir"] = osp.join(slurm_USERDIR, cfg["data"]["train"]["root_dir"])
        cfg["data"]["val"]["root_dir"] = osp.join(slurm_USERDIR, cfg["data"]["val"]["root_dir"])
        world_size = int(os.environ['SLURM_NTASKS'])
        num_gpus = world_size
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["true_gpu"]
        output_dir = cfg["output_dir"]
        num_gpus = len(cfg["gpu"])
        world_size = num_gpus

    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime(".%m_%d_%H_%M_%S")
    log_dir = os.path.join(output_dir, "log{}".format(timestamp))

    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    cfg["log_dir"] = log_dir

    # copy config file to log_dir
    shutil.copy(args.config_file, log_dir)

    logger = setup_logger("gbinet", log_dir, prefix="train")
    try:
        git_commit_id = \
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[0:-1]
        git_branch_name = \
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('UTF-8')[0:-1]
    except:
        logger.info("No git founded")
        git_commit_id = ""
        git_branch_name = ""

    logger.info("Branch " + git_branch_name)
    logger.info("Commit ID " + git_commit_id)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(sys.argv))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    cfg["world_size"] = world_size
    if cfg.get("slurm", False):
        train(-1, cfg)
    else:
        mp.spawn(train,
            args=(cfg,),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    main()
