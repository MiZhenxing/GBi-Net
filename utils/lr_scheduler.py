import torch

def build_scheduler(cfg, optimizer, n_iter_per_epoch):

    lr_scheduler = None
    if cfg["scheduler"]["name"] == 'multi_step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["scheduler"]["milestones"], 
            gamma=cfg["scheduler"]["gamma"]
        )    
    elif cfg["scheduler"]["name"] == 'none':
        lr_scheduler = None

    return lr_scheduler
