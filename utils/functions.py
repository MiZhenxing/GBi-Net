import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import random
import torch.distributed as dist
from matplotlib import cm
import matplotlib.pyplot as plt

# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# a wrapper to compute metrics for each image individually
def compute_batch_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, thres):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], thres[idx])
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_batch_metrics_for_each_image
def Batch_Thres_metrics(depth_est, depth_gt, mask, thres):
    # assert isinstance(thres, (int, float))
    if torch.all(torch.logical_not(mask)):
        error = torch.tensor(0.0, device=depth_est.device)
    else:
        depth_est, depth_gt = depth_est[mask], depth_gt[mask]
        errors = torch.abs(depth_est - depth_gt)
        err_mask = errors > thres
        error = torch.mean(err_mask.float())
    return error


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask):
    if torch.all(torch.logical_not(mask)):
        error = torch.tensor(0.0, device=depth_est.device)
    else:
        depth_est, depth_gt = depth_est[mask], depth_gt[mask]
        error = torch.mean((depth_est - depth_gt).abs())
    return error


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# metric from MVSNet_pl
def abs_error_pl(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return (depth_pred - depth_gt).abs()

# metric from MVSNet_pl
def acc_threshold_pl(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error_pl(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.float().mean()

@make_nograd_func
@compute_metrics_for_each_image
def GBiNet_accu(pred_label, gt_label, mask):
    # pred_label: B, H, W
    # gt_label: B, H, W
    # mask: B, H, W
    if torch.all(torch.logical_not(mask)):
        accu = torch.tensor(0.0, device=pred_label.device)
    else:
        accu = torch.mean(torch.eq(pred_label[mask], gt_label[mask]).float())
    return accu

# NOTE: please do not use this to build up training loss
@make_nograd_func
def Prob_mean(prob_map, mask):
    batch_size = prob_map.shape[0]
    results = []
    # compute result one by one
    for idx in range(batch_size):
        ret = torch.mean(prob_map[idx][mask[idx]])
        results.append(ret)
    return torch.stack(results).mean()

@make_nograd_func
def mapping_color(img, vmin, vmax, cmap="rainbow"):
    batch_size = img.shape[0] # B, H, W
    results = []
    # compute result one by one
    for idx in range(batch_size):
        np_img = img[idx].cpu().numpy()
        if vmin is not None and vmax is not None:
            np_img = plt.Normalize(vmin=vmin[idx].item(), vmax=vmax[idx].item())(np_img)
        mapped_img = getattr(cm, cmap)(np_img)
        results.append(mapped_img[:, :, 0:4])
    results = torch.tensor(np.stack(results), device=img.device)
    results = results.permute(0, 3, 1, 2)
    return results

def chunk_list(L, n=1, verbose=False):
    '''
    Partition list L into n chunks.
    
    Returns a list of n lists/chunks, where each chunk is 
    of nearly equal size.
        >>> L = 'a b c d'.split(" ")
        ['a', 'b', 'c', 'd']
        >>> chunk(L, 2)
        [['a', 'b'], ['c', 'd']]
        >>> chunk(L, 3)
        [['a', 'b'], ['c'], ['d']]
        >>> chunk(L, 4)
        [['a'], ['b'], ['c'], ['d']]
        >>> chunk(L, 5)
        [['a'], ['b'], ['c'], ['d'], []]
    '''
    total = len(L)
    if n > 0:
        size = total // n
        rest = total % n
        ranges = []
        if verbose:
            msg = "{} items to be split into {} chunks of size {} with {} extra"
            print(msg.format(total, n, size, rest))
        if not size:
            return [[x] for x in L] + [[] for i in range(n - total)]
        if rest:
            index = [x for x in range(0, total, size)]
            extra = [index[i] + i for i in range(rest + 1)] + \
                    [x + rest for x in index[rest+1:][:n-rest]]
            ranges = [(extra[i], extra[i+1]) for i in range(len(extra) - 1)]
        else:
            index = [x for x in range(0, total+1, size)]
            ranges = [(index[i], index[i+1]) for i in range(len(index) - 1)]
        return [L[i:j] for i, j in ranges]
