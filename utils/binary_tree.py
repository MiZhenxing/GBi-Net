import torch

def update_tree1(binary_tree, indicator, direction):
    r"""update the tree to the next or last level, or keep to the current level

    Args:
        indicator (integer tensor): indicator tensor for updating. 
            -1 means going up and 1 means going down, 
            while 0 means staying.
        direction (positive integer tensor): direction of moving to left or right.
            0 means left, while 1 means right. If the values of indicator are not 1, 
            the correspoding values of direction must be 0.
    """
    tree_list = []
    tree_list.append(torch.clamp_min(binary_tree[:, 0, :, :] + torch.squeeze(indicator, 1), 0))
    tree_key = torch.clamp_min((binary_tree[:, 1, :, :] * \
        (2.0 ** torch.squeeze(indicator, 1))).type(torch.int64) + \
        torch.squeeze(direction, 1), 0)
    tree_key = torch.minimum(tree_key, 2 ** tree_list[0] - 1) # clamp max, should be 2 ** tree_list[0] - 1
    tree_list.append(tree_key)
    updated_tree = torch.stack(tree_list, 1)
    return updated_tree