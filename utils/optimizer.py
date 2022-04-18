# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(cfg, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    # skip = {}
    # skip_keywords = {}
    # if hasattr(model, 'no_weight_decay'):
    #     skip = model.no_weight_decay()
    # if hasattr(model, 'no_weight_decay_keywords'):
    #     skip_keywords = model.no_weight_decay_keywords()
    # parameters = set_weight_decay(model, skip, skip_keywords)
    parameters = model.parameters()

    opt_lower = cfg["solver"]["name"].lower()
    optimizer = None
    if not cfg["solver"].get("group", False):
        if opt_lower == 'sgd':
            optimizer = optim.SGD(parameters, momentum=cfg["solver"]["momentum"], nesterov=True,
                                lr=cfg["solver"]["base_lr"], weight_decay=cfg["solver"]["weight_decay"])
        elif opt_lower == 'adamw':
            optimizer = optim.AdamW(parameters, eps=cfg["solver"]["eps"], betas=cfg["solver"]["betas"],
                                    lr=cfg["solver"]["base_lr"], weight_decay=cfg["solver"]["weight_decay"])
        elif opt_lower == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg["solver"]["base_lr"], 
                betas=cfg["solver"]["betas"], weight_decay=cfg["solver"]["weight_decay"])
    else:
        if opt_lower == 'sgd':
            optimizer = optim.SGD([{'params': model.module.feature.parameters(), 
                                    "momentum": cfg["solver"]["feature"]["momentum"], "nesterov": True,
                                    "lr": cfg["solver"]["feature"]["base_lr"], "weight_decay": cfg["solver"]["feature"]["weight_decay"]},
                                    {'params': model.module.cost_regularization.parameters()}], 
                                    momentum=cfg["solver"]["momentum"], nesterov=True,
                                    lr=cfg["solver"]["base_lr"], weight_decay=cfg["solver"]["weight_decay"])
        elif opt_lower == 'adamw':
            optimizer = optim.AdamW([{'params': model.module.feature.parameters(), 
                                    "eps": cfg["solver"]["feature"]["eps"], "betas": cfg["solver"]["feature"]["betas"],
                                    "lr": cfg["solver"]["feature"]["base_lr"], "weight_decay": cfg["solver"]["feature"]["weight_decay"]},
                                    {'params': model.module.cost_regularization.parameters()}], 
                                    eps=cfg["solver"]["eps"], betas=cfg["solver"]["betas"],
                                    lr=cfg["solver"]["base_lr"], weight_decay=cfg["solver"]["weight_decay"])
        elif opt_lower == 'adam':
            optimizer = optim.Adam([{'params': model.module.feature.parameters(), 
                                    "lr": cfg["solver"]["feature"]["base_lr"], 
                                    "betas": cfg["solver"]["feature"]["betas"], "weight_decay": cfg["solver"]["feature"]["weight_decay"]},
                                    {'params': model.module.cost_regularization.parameters()}], 
                                    lr=cfg["solver"]["base_lr"], 
                                    betas=cfg["solver"]["betas"], weight_decay=cfg["solver"]["weight_decay"])

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
