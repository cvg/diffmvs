import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")
x = torch.inverse(torch.ones((1, 1), device="cuda:0"))

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class SepConvGRU(nn.Module):
    """Separable convolutional GRU from RAFT"""
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

def differentiable_warping(src_fea, src_proj, ref_proj, depth_values):
    """get warped source image features"""
    B = src_fea.shape[0]
    num_depth, height0, width0 = depth_values.shape[1:]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid(
            [torch.arange(0, height0, dtype=torch.float32, device=src_fea.device),
            torch.arange(0, width0, dtype=torch.float32, device=src_fea.device)]
        )
        y, x = y.contiguous(), x.contiguous()

        y, x = y.view(height0 * width0), x.view(height0 * width0)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = (rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * 
                        depth_values.view(B, 1, num_depth, -1))

        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)
        proj_xyz[:, 2:3][proj_xyz[:, 2:3] == 0] += 1e-8
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)

    warped_src_fea = F.grid_sample(
        src_fea, proj_xy.view(B, num_depth * height0, width0, 2),
        mode='bilinear', padding_mode='zeros', align_corners=True
    )

    warped_src_fea = warped_src_fea.view(B, -1, num_depth, height0, width0)
    return warped_src_fea

def disp_to_depth(disp, min_depth, max_depth):
    """transform normalized inverse depth to metric depth"""
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    scaled_disp = scaled_disp.clamp(min = 1e-6)
    depth = 1 / scaled_disp
    return scaled_disp, depth

def depth_to_disp(depth, min_depth, max_depth):
    """transform metric depth ro normalized inverse depth"""
    scaled_disp = 1 / depth
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = (scaled_disp - min_disp) / ((max_disp - min_disp))
    return disp

def upsample_depth(depth, mask, ratio = 8):
    """upsample depth map using convex combination"""
    N, _, H, W = depth.shape
    mask = mask.view(N, 1, 9, ratio, ratio, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(depth, [3, 3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, ratio * H, ratio * W)

def get_cur_depth_range_samples(
        cur_depth,
        ndepth,
        depth_inteval_pixel,
        confidence=None,
        min=0.2,
        max=2
    ):
    """sample new depth hypotheses in the inverse range"""
    if confidence is None:
        cur_depth_min = (cur_depth - ndepth // 2 * depth_inteval_pixel)
        cur_depth_max = (cur_depth + ndepth // 2 * depth_inteval_pixel)
    else:
        radius = ndepth // 2 * depth_inteval_pixel
        radius_min = min * radius
        radius_max = max * radius
        radius = radius_min + (1 - confidence) * (radius_max - radius_min)
        cur_depth_min = (cur_depth - radius)
        cur_depth_max = (cur_depth + radius)

    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)
    depth_range_samples = torch.arange(0, ndepth, device=cur_depth.device,
        dtype=cur_depth.dtype,
        requires_grad=False
    ).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1)
    depth_range_samples += cur_depth_min.unsqueeze(1)
    depth_range_samples = torch.clamp(depth_range_samples, min=0, max=1)
    return depth_range_samples

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_planes, planes, 3, stride=stride, pad=1)
        self.conv2 = ConvBn(planes, planes, 3, stride=1, pad=1)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = ConvBn(in_planes, planes, 3, stride=stride, pad=1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)
    
class ContextNet(nn.Module):
    """context feature extraction of reference image"""
    def __init__(self, out_dim=[16,16,16]):
        super(ContextNet, self).__init__()
        self.in_planes = 8
        self.out_dim = out_dim
        self.conv1 = ConvBnReLU(3,8)
        self.layer1 = self._make_layer(16, stride=2)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(48, stride=2)

        self.output1 = nn.Conv2d(48, out_dim[0], 3, stride=1, padding=1)
        self.output2 = nn.Conv2d(32, out_dim[1], 3, stride=1, padding=1)
        if out_dim[2] > 0:
            # for CasDiffMVS
            self.output3 = nn.Conv2d(16, out_dim[2], 3, stride=1, padding=1)

    def _make_layer(self, dim, stride=1):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        context = {}
        x = self.conv1(x)
        x = self.layer1(x)
        if self.out_dim[2] > 0:
            context['stage3'] = self.output3(x)
        x = self.layer2(x)
        context['stage2'] = self.output2(x)
        x = self.layer3(x)
        context['stage1'] = self.output1(x)
        return context

class FeatureNet(nn.Module):
    """image feature extraction"""
    def __init__(self, base_channels=8, out_channel=[32,16,8]):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels
        self.out_channel = out_channel

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )
        
        self.out1 = nn.Conv2d(base_channels * 8, out_channel[0], 1, bias=False)
        
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.out2 = nn.Conv2d(final_chs, out_channel[1], 3, padding=1, bias=False)

        if out_channel[2]>0:
            # for CasDiffMVS
            self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
            self.out3 = nn.Conv2d(final_chs, out_channel[2], 3, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        intra_feat = conv3
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2,
                                   mode="nearest") + self.inner1(conv2)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        if self.out_channel[2] > 0:
            intra_feat = F.interpolate(intra_feat, scale_factor=2,
                                       mode="nearest") + self.inner2(conv1)
            out = self.out3(intra_feat)
            outputs["stage3"] = out

        return outputs

class CostRegNet_small(nn.Module):
    """3D cost volume regularization"""
    def __init__(self, in_channels, base_channels):
        super(CostRegNet_small, self).__init__()

        self.conv0 = Conv3d(in_channels, base_channels, padding=1)
        self.conv1 = Conv3d(base_channels, base_channels, padding=1)

        self.conv2 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv3 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv4 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv5 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv6 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)
        self.conv7 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv1 = self.conv1(self.conv0(x))
        conv3 = self.conv3(self.conv2(conv1))
        x = self.conv5(self.conv4(conv3))
        x = conv3 + self.conv6(x)
        x = conv1 + self.conv7(x)
        x = self.prob(x)
        return x

class PixelViewWeight(nn.Module):
    """Estimate pixel-wise view weight"""
    def __init__(self, G):
        super(PixelViewWeight, self).__init__()
        self.conv = nn.Sequential(
            Conv3d(G, 8, padding=1),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x).squeeze(1)
        x = torch.sigmoid(x)
        x = torch.max(x, dim=1)[0]
        return x.unsqueeze(1)

class InitialCost(nn.Module):
    """Cost volume construction in depth initialization"""
    def __init__(self, feature_dim, group_dim = 8, ratio = 2):
        """
        Params:
            feature_dim: dimension of image feature
            group_dim: feature dimension of each feature group
            ratio: upsample ratio of depth map
        """
        super(InitialCost, self).__init__()
        self.group_dim = group_dim
        self.pixel_view_weight = PixelViewWeight(self.group_dim)
        self.cost_regularization = CostRegNet_small(
            in_channels=group_dim, base_channels=8
        )
        # mask for upsampling
        self.mask = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, ratio*ratio*9, 1, padding=0)
        )

    def forward(
        self,
        features,
        context,
        proj_matrices,
        depth_values,
        scale_inv_depth=None
    ):
        """
        Params:
            features: image features
            context: context feature of reference image
            proj_matrices: camera projection matrices
            depth_values: depth samples
            scale_inv_depth: transform inverse depth to metric depth
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        num_depth = depth_values.size(1)

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B,D,H,W = depth_values.shape
        C = ref_feature.shape[1]

        mask = .25 * self.mask(context) # mask for upsampling depth map

        # cost volume construction between reference image and each source image
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        view_weight_sum = 1e-8
        cor_feats = 0
        view_weights = []
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3],
                                                   src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3],
                                                   ref_proj[:, 0, :3, :4])
            warped_src = differentiable_warping(src_fea, src_proj_new,
                                                ref_proj_new, depth_values)

            warped_src = warped_src.reshape(B, self.group_dim, C//self.group_dim, D, H, W)
            ref_volume = ref_volume.reshape(B, self.group_dim, C//self.group_dim, D, H, W)
            cor_feat = (warped_src * ref_volume).mean(2)

            view_weight = self.pixel_view_weight(cor_feat)

            del warped_src, src_proj, src_fea

            view_weights.append(view_weight)

            if self.training:
                view_weight_sum = view_weight_sum + view_weight.unsqueeze(1)
                cor_feats = cor_feats + view_weight.unsqueeze(1) * cor_feat
            else:
                view_weight_sum += view_weight.unsqueeze(1)
                cor_feats += view_weight.unsqueeze(1) * cor_feat
            del view_weight, cor_feat
        
        # aggregated matching cost across all the source views
        cor_feats = cor_feats.div_(view_weight_sum)
        view_weights = torch.cat(view_weights,dim=1)
        del view_weight_sum, src_features

        # cost volume regularization
        prob_volume_pre = self.cost_regularization(cor_feats).squeeze(1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)

        index = torch.arange(0, num_depth, 1,
                             device=prob_volume.device).view(1, num_depth, 1, 1).float()
        index = torch.sum(index * prob_volume, dim = 1, keepdim=True) # [B,1,H,W]
        normalized_depth = index / (num_depth-1.0)
        depth = (scale_inv_depth(normalized_depth)[1]).squeeze(1)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(
                prob_volume.unsqueeze(1),
                pad=(0, 0, 0, 0, 1, 2)
            ), (4, 1, 1), stride=1, padding=0).squeeze(1)

            index = index.long()
            index = index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, index)

        return mask, normalized_depth, depth, view_weights.detach(), photometric_confidence

class GetCost(nn.Module):
    """compute local cost volume"""
    def __init__(self, group_dim=4, min_radius=0.2, max_radius=2):
        super(GetCost, self).__init__()
        self.group_dim = group_dim
        self.min_radius = min_radius
        self.max_radius = max_radius

    def forward(
        self, 
        inverse_depth, 
        features, 
        proj_matrices, 
        depth_interval, 
        depth_max, 
        depth_min, 
        CostNum = 4, 
        view_weights = None,
        confidence = None,
    ):
        """
        Params:
            features: image features
            proj_matrices: camera projection matrices
            inverse_depth: current inverse depth map
            depth_interval: interval for sampling depth values
            depth_max: maximum depth value
            depth_min: minimum depth value
            CostNum: number of new samples
            view_weights: pixel-wise view weight
            confidence: confidence from previous iteration if exists
        """
        proj_matrices = torch.unbind(proj_matrices, 1)

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        if CostNum > 1:
            inverse_depth_samples = get_cur_depth_range_samples(
                cur_depth=inverse_depth.squeeze(1),
                ndepth=CostNum,
                depth_inteval_pixel=depth_interval,
                confidence=confidence,
                min=self.min_radius,
                max=self.max_radius,
            )
        else:
            inverse_depth_samples = inverse_depth

        depth_range_samples = disp_to_depth(inverse_depth_samples, depth_min, depth_max)[1]

        B,D,H,W = depth_range_samples.shape
        C = ref_feature.shape[1]
 
        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, depth_range_samples.shape[1], 1, 1)
        view_weight_sum = 1e-8
        cor_feats = 0
        i = 0
        for src_fea, src_proj in zip(src_features, src_projs):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3],
                                                   src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3],
                                                   ref_proj[:, 0, :3, :4])
            warped_src = differentiable_warping(src_fea, src_proj_new,
                                                ref_proj_new, depth_range_samples)

            warped_src = warped_src.reshape(B, self.group_dim, C//self.group_dim, D, H, W)
            ref_volume = ref_volume.reshape(B, self.group_dim, C//self.group_dim, D, H, W)
            cor_feat = (warped_src * ref_volume).mean(2)
            
            del warped_src, src_proj, src_fea

            view_weight = view_weights[:, i].unsqueeze(1)
            i = i + 1

            if self.training:
                view_weight_sum = view_weight_sum + view_weight.unsqueeze(1)
                cor_feats = cor_feats + view_weight.unsqueeze(1) * cor_feat
            else:
                view_weight_sum += view_weight.unsqueeze(1)
                cor_feats += view_weight.unsqueeze(1) * cor_feat
            del view_weight, cor_feat

        cor_feats = cor_feats.div_(view_weight_sum)

        del view_weight_sum, src_features

        b,c,d,h,w = cor_feats.shape
        cor_feats = cor_feats.view(b, c*d, h, w)
        return cor_feats, inverse_depth_samples
