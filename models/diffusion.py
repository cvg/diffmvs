import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

from .module import *
from .update import DiffusionUpdateBlockDepth

class CasDiffMVS(nn.Module):
    """Implementation of DiffMVS and CasDiffMVS"""

    def __init__(
        self,
        args,
        depth_interals_ratio=[4,2,1],
        test=False
    ):
        """
        Params:
            args: arguments from parser
            depth_interals_ratio: sampling interval of inverse depth
            test: denote training or testing
        """
        super(CasDiffMVS, self).__init__()

        self.numdepth_initial = args.numdepth_initial
        self.depth_interals_ratio = depth_interals_ratio
        self.args = args
        self.num_stage = 3
        self.cost_dim_stage = args.cost_dim_stage

        self.unet_dim = args.unet_dim
        self.unet_dim_mults = [(1,), (1,2), (1,2,4)]
        self.test = test

        if args.stage_iters[2] == 0:
            """ DiffMVS: single stage refinement """
            self.up_ratio = 4  # upsample from 1/4 resolution to full resolution
            self.CostNum = args.CostNum
            self.feat_dim_stage = [48, 32, 0]
            self.hdim_stage = args.hidden_dim
            self.cdim_stage = args.context_dim
            self.context_dim = [self.hdim_stage[0]+self.cdim_stage[0],
                                self.hdim_stage[1]+self.cdim_stage[1],
                                self.hdim_stage[2]+self.cdim_stage[2]]

            self.feature = FeatureNet(
                base_channels=8, 
                out_channel=self.feat_dim_stage, 
            )
            self.context = ContextNet(self.context_dim)

            self.hidden_init = nn.ModuleList([
                nn.Sequential(
                    Conv2d(self.hdim_stage[1], 32, 3, 2, padding=1),
                    nn.Conv2d(32, self.hdim_stage[1], 3, 1, padding=1, bias=False),
                ),
            ])
            self.update_block_depth2 = DiffusionUpdateBlockDepth(
                args, 
                dim=self.unet_dim[1], 
                dim_mults=self.unet_dim_mults[1], 
                hidden_dim=self.hdim_stage[1], 
                num_sample=self.CostNum[1],
                cost_dim=self.cost_dim_stage[1]*self.CostNum[1],
                context_dim=self.cdim_stage[1], 
                stage_idx=1, 
                iters=args.stage_iters[1],
                ratio=self.up_ratio,
            )
            self.update_block = nn.ModuleList([self.update_block_depth2,])
            
        else:
            """ CasDiffMVS: multi stage refinement """
            self.up_ratio = 2  # upsample from 1/2 resolution to full resolution
            self.CostNum = args.CostNum
            self.feat_dim_stage = [48, 32, 16]
            self.hdim_stage = args.hidden_dim
            self.cdim_stage = args.context_dim
            self.context_dim = [self.hdim_stage[0]+self.cdim_stage[0],
                                self.hdim_stage[1]+self.cdim_stage[1],
                                self.hdim_stage[2]+self.cdim_stage[2]]

            self.feature = FeatureNet(
                base_channels=8, 
                out_channel=self.feat_dim_stage, 
            )
            self.context = ContextNet(self.context_dim)

            # need downsampling
            self.hidden_init = nn.ModuleList([
                nn.Sequential(
                    Conv2d(self.hdim_stage[1], 32, 3, 2, padding=1),
                    nn.Conv2d(32, self.hdim_stage[1], 3, 1, padding=1, bias=False),
                ),
                nn.Sequential(
                    Conv2d(self.hdim_stage[2], 32, 3, 2, padding=1),
                    Conv2d(32, 32, 3, 2, padding=1),
                    nn.Conv2d(32, self.hdim_stage[2], 3, 1, padding=1, bias=False),
                )
            ])
            self.update_block_depth2 = DiffusionUpdateBlockDepth(
                args, 
                dim=self.unet_dim[1], 
                dim_mults=self.unet_dim_mults[1], 
                hidden_dim=self.hdim_stage[1], 
                num_sample=self.CostNum[1],
                cost_dim=self.cost_dim_stage[1]*self.CostNum[1],
                context_dim=self.cdim_stage[1], 
                stage_idx=1, 
                iters=args.stage_iters[1],
                ratio=self.up_ratio,
            )

            self.update_block_depth3 = DiffusionUpdateBlockDepth(
                args, 
                dim=self.unet_dim[2], 
                dim_mults=self.unet_dim_mults[2], 
                hidden_dim=self.hdim_stage[2], 
                num_sample=self.CostNum[2],
                cost_dim=self.cost_dim_stage[2]*self.CostNum[2],
                context_dim=self.cdim_stage[2], 
                stage_idx=2, 
                iters=args.stage_iters[2],
                ratio=self.up_ratio,
            )

            self.update_block = nn.ModuleList([self.update_block_depth2,
                                               self.update_block_depth3])

        self.depthnet = InitialCost(self.cdim_stage[0], self.cost_dim_stage[0])
        self.GetCost = GetCost(
            self.cost_dim_stage[1],
            min_radius=args.min_radius,
            max_radius=args.max_radius
        )


    def forward(self, imgs, proj_matrices, depth_values, depth_gt_ms = None):
        disp_min = depth_values[:, 0].float().view(-1,1,1,1)
        disp_max = depth_values[:, -1].float().view(-1,1,1,1)
        depth_max_ = 1. / disp_min
        depth_min_ = 1. / disp_max
        depth_interval = 1.0 / depth_values.size(1)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=depth_min_, max_depth=depth_max_)

        features = []       # store image features
        confs = []          # store confidence maps with different sizes
        confidences = []    # store confidence maps with full resoultion
        depth_predictions = []

        """Feature Extraction"""
        # torch.cuda.synchronize()
        # start_time = time.time()
        for img in imgs:
            features.append(self.feature(img))

        contexts = self.context(imgs[0])
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('feature, Time:{}'.format(end_time - start_time))

        """Depth estimation in multiple stages"""
        for stage_idx in range(self.num_stage):
            # torch.cuda.synchronize()
            # start_time = time.time()
            if self.training and stage_idx > 0:
                # convert ground truth depth to disparity
                depth_gt_stage = depth_gt_ms[f"stage{stage_idx+1}"].unsqueeze(1)
                _, _, H, W = depth_gt_stage.size()
                inv_depth_gt = depth_to_disp(depth_gt_stage, depth_min_, depth_max_)
            else:
                inv_depth_gt = None
            
            # DiffMVS has no refinement at 1/2 resolution
            if self.args.stage_iters[stage_idx] == 0:
                continue

            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)].float()
            ref_feature = features_stage[0]
            context_stage = contexts["stage{}".format(stage_idx + 1)]
            B, _, H, W = ref_feature.size()
            if stage_idx == 0:
                """Depth Initialization at 1/8 resolution"""
                depth_range_samples = torch.arange(0, self.numdepth_initial,
                                                   device=ref_feature.device).view(1, -1, 1, 1)
                depth_range_samples = depth_range_samples / (self.numdepth_initial - 1.0)

                depth_range_samples = depth_range_samples.repeat(1, 1, H, W)
                depth_range_samples = self.scale_inv_depth(depth_range_samples)[1]

                context = torch.relu(context_stage)

                mask, inv_depth, init_depth, view_weights, conf = self.depthnet(
                    features_stage,
                    context,
                    proj_matrices_stage,
                    depth_values=depth_range_samples,
                    scale_inv_depth=self.scale_inv_depth
                )

                depth_predictions.append(init_depth)
                confidences.append(F.interpolate(conf,
                                        scale_factor=2**(3-stage_idx), 
                                        mode='nearest').squeeze(1))

                inv_depth_up = upsample_depth(inv_depth, mask, ratio=2).unsqueeze(1)
                final_depth = self.scale_inv_depth(inv_depth_up)[1].squeeze(1)
                depth_predictions.append(final_depth)

            else:
                """Diffusion-based refinement"""
                cur_depth = depth_predictions[-1].unsqueeze(1)
                cur_depth = cur_depth.detach()
                inv_cur_depth = depth_to_disp(cur_depth, depth_min_, depth_max_)

                view_weights_stage = F.interpolate(view_weights,
                                                   scale_factor=2 ** (stage_idx),
                                                   mode='nearest')

                hidden_d, context = torch.split(
                    context_stage,
                    [self.hdim_stage[stage_idx], self.cdim_stage[stage_idx]],
                    dim=1
                )
                # initialize hidden state
                hidden_d = self.hidden_init[stage_idx-1](hidden_d)
                current_hidden_d = torch.tanh(hidden_d)
                context = torch.relu(context)
                
                if not self.test:
                    # pseudo gt from initial depth map for the empty areas without gt
                    init_depth_1 = F.interpolate(init_depth.unsqueeze(1), 
                                                 scale_factor=2**stage_idx, mode='nearest')
                    inv_init_depth = depth_to_disp(init_depth_1, depth_min_, depth_max_)
                    inv_init_depth = inv_init_depth.detach()
                else:
                    inv_init_depth = None

                depth_cost_func = partial(
                    self.GetCost,
                    features=features_stage,
                    proj_matrices=proj_matrices_stage,
                    depth_interval=depth_interval*self.depth_interals_ratio[stage_idx],
                    depth_max=depth_max_,
                    depth_min=depth_min_,
                    CostNum=self.CostNum[stage_idx],
                    view_weights=view_weights_stage,
                )

                mask, current_hidden_d, inv_depth_seqs, conf_seqs = self.update_block[stage_idx-1](
                    depth_cost_func,
                    inv_cur_depth,
                    current_hidden_d,
                    context,
                    gt_inv_depth=inv_depth_gt,
                    inv_init_depth=inv_init_depth,
                )

                if not self.test:
                    # during training, store all intermediate results for loss computation
                    for inv_depth_i in inv_depth_seqs:
                        depth_predictions.append(
                            self.scale_inv_depth(inv_depth_i)[1].squeeze(1)
                        )
                    for conf in conf_seqs:
                        confs.append(conf)
                else:
                    depth_predictions.append(
                        self.scale_inv_depth(inv_depth_seqs[-1])[1].squeeze(1)
                    )
                    confidences.append(
                        F.interpolate(conf_seqs[-1].unsqueeze(1),
                                        scale_factor=2**(3-stage_idx), 
                                        mode='nearest').squeeze(1)
                    )
                
                last_inv_depth = inv_depth_seqs[-1]
                inv_depth_up = upsample_depth(
                    last_inv_depth, mask, ratio=self.up_ratio
                ).unsqueeze(1)
                final_depth = self.scale_inv_depth(inv_depth_up)[1].squeeze(1)
                depth_predictions.append(final_depth)
            
            # torch.cuda.synchronize()
            # end_time = time.time()
            # print('stage {}, Time:{}'.format(stage_idx + 1, end_time - start_time))
        
        return {
                "depth": depth_predictions, 
                "conf": confs,
                "photometric_confidence": confidences, 
                }
