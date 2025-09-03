import torch
import numpy as np
import torch.nn.functional as F
from .module import depth_to_disp

def compute_inverse_loss(
    args, 
    inputs, 
    confs,
    depth_gt_ms, 
    mask_ms, 
    depth_values, 
    loss_rate=0.8,
    iters=[1,3,3],
):
    total_loss = 0.0
    loss_dict = {}
    loss_len = len(inputs)
    loss_rate = loss_rate
    if iters[2] == 0:
        # DiffMVS
        stage_id = ([1 for i in range(iters[0])] +
                    [2 for i in range(iters[1]+1)] + [4])

        conf_flag = ([False for i in range(iters[0]+1)] +
                     [True for i in range(iters[1])] + [False])
    else:
        # CasDiffMVS
        stage_id = ([1 for i in range(iters[0])] +
                    [2 for i in range(iters[1]+1)] +
                    [3 for i in range(iters[2]+1)] + [4])

        conf_flag = ([False for i in range(iters[0] + 1)] +
                     [True for i in range(iters[1])] + [False] +
                     [True for i in range(iters[2])] + [False])

    assert loss_len == len(stage_id), "input depths need to have the same number as stage_id."

    disp_min = depth_values[:, 0, None, None]
    disp_max = depth_values[:, -1, None, None]
    depth_max = 1. / disp_min
    depth_min = 1. / disp_max
    
    conf_iter = 0
    for i, depth_esti in enumerate(inputs):
        depth_est = depth_to_disp(depth_esti, depth_min, depth_max)

        # transform ground truth depth to normalized inverse depth
        depth_gt = depth_gt_ms["stage{}".format(stage_id[i])]
        B,H,W = depth_gt.size()
        depth_maxs = depth_max.view(-1,1,1).repeat(1,H,W)
        depth_gt = torch.where(depth_gt>1e-4, depth_gt, depth_maxs)
        depth_gt = depth_to_disp(depth_gt, depth_min, depth_max)

        mask = mask_ms["stage{}".format(stage_id[i])]
        mask = mask > 0.5

        if conf_flag[i]:
            # there is estimated confidence from diffusion model
            confidence = confs[conf_iter]
            conf_iter = conf_iter + 1
            uncertainty = 1 - confidence
            uncertainty = torch.clamp(uncertainty, min=1e-6)
            depth_loss = torch.abs(depth_est - depth_gt)
            depth_loss = depth_loss / uncertainty + args.conf_weight * torch.log(uncertainty)
            depth_loss = torch.mean(depth_loss[mask])
        else:
            depth_loss = F.l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        loss_dict["l{}".format(i)] = F.l1_loss(depth_est[mask],
                                               depth_gt[mask], reduction='mean')
        loss_weight = loss_rate ** (loss_len - i - 1) # exponentially increase weight
        total_loss += loss_weight * depth_loss

    return total_loss, loss_dict