import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.deform_conv import DeformConv2dPack


class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, in_affine=True, num_groups=8, **kwargs):
        super(Conv2dUnit, self).__init__()
        bias = False
        if norm is None:
            bias = True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm2d(out_channels, momentum=in_momentum, affine=in_affine)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
                 in_affine=True, num_groups=8, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride
        bias = False
        if norm is None:
            bias = True
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=bias, **kwargs)
        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm2d(out_channels, momentum=in_momentum, affine=in_affine)
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.norm_layer is not None:
            y = self.norm_layer(y)
        if self.relu:
            y = F.relu(y, inplace=True)
        return y

class Conv3dUnit(nn.Module):
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
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
                 in_affine=True, num_groups=8, init_method="xavier", **kwargs):
        super(Conv3dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # assert stride in [1, 2]
        self.stride = stride
        bias = False
        if norm is None:
            bias = True

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, **kwargs)
        
        
        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm3d(out_channels, momentum=in_momentum, affine=in_affine)

        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv3dUnit(nn.Module):
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
                 relu=True, norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
                 in_affine=True, num_groups=8, init_method="xavier", **kwargs):
        super(Deconv3dUnit, self).__init__()
        self.out_channels = out_channels
        # assert stride in [1, 2]
        self.stride = stride
        bias = False
        if norm is None:
            bias = True

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=bias, **kwargs)

        self.norm_layer = None
        if norm == "batchnorm":
            self.norm_layer = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        if norm == "groupnorm":
            self.norm_layer = nn.GroupNorm(num_groups, out_channels)
        if norm == "instancenorm":
            self.norm_layer = nn.InstanceNorm3d(out_channels, momentum=in_momentum, affine=in_affine)

        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.norm_layer is not None:
            y = self.norm_layer(y)
        if self.relu:
            y = F.relu(y, inplace=True)
        return y

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, 
        norm="batchnorm", bn_momentum=0.1, in_momentum=0.1, 
        in_affine=True, num_groups=8):
        super(Deconv2dBlock, self).__init__()

        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
            relu=relu, norm=norm, bn_momentum=bn_momentum, in_momentum=in_momentum, 
            in_affine=in_affine, num_groups=num_groups)

        self.conv = Conv2dUnit(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
            relu=relu, norm=norm, bn_momentum=bn_momentum, in_momentum=in_momentum, 
            in_affine=in_affine, num_groups=num_groups)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class FeatExtNet(nn.Module):
    def __init__(self, in_channel, base_channels):
        super(FeatExtNet, self).__init__()

        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2dUnit(in_channel, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)
        self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3)

        self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        intra_feat = conv2
        intra_feat = self.deconv1(conv1, intra_feat)
        intra_feat = self.deconv2(conv0, intra_feat)
        out = self.out3(intra_feat)

        return out

class StageFeatExtNet(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None):
        super(StageFeatExtNet, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [32, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num
        # [B,8,H,W]
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        # [B,16,H/2,W/2]
        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        # [B,32,H/4,W/4]
        self.conv2 = nn.Sequential(
                Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = nn.Sequential(
            Conv2dUnit(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            )

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = nn.Conv2d(base_channels * 8, self.output_channels[0], 1, bias=False)
            self.conv_out["1"] = nn.Conv2d(base_channels * 8, self.output_channels[1], 1, bias=False)
            self.conv_out["2"] = nn.Conv2d(base_channels * 4, self.output_channels[2], 1, bias=False)
            self.conv_out["3"] = nn.Conv2d(base_channels * 2, self.output_channels[3], 1, bias=False)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 8, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["3"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)
        
        if stage_num == 3:
            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = nn.Conv2d(base_channels * 4, self.output_channels[0], 1, bias=False)
            self.conv_out["1"] = nn.Conv2d(base_channels * 4, self.output_channels[1], 1, bias=False)
            self.conv_out["2"] = nn.Conv2d(base_channels * 2, self.output_channels[2], 1, bias=False)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)

    def forward(self, x):
        output_feature = {}
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)
            output_feature["0"] = self.conv_out["0"](conv3)
            intra_feat = F.interpolate(conv3, scale_factor=2, mode="bilinear") + self.conv_inner["1"](conv2)
            output_feature["1"] = self.conv_out["1"](intra_feat)
            intra_feat = F.interpolate(conv2, scale_factor=2, mode="bilinear") + self.conv_inner["2"](conv1)
            output_feature["2"] = self.conv_out["2"](intra_feat)
            intra_feat = F.interpolate(conv1, scale_factor=2, mode="bilinear") + self.conv_inner["3"](conv0)
            output_feature["3"] = self.conv_out["3"](intra_feat)

        if self.stage_num == 3:
            output_feature["0"] = self.conv_out["0"](conv2)
            intra_feat = F.interpolate(conv2, scale_factor=2, mode="bilinear") + self.conv_inner["1"](conv1)
            output_feature["1"] = self.conv_out["1"](intra_feat)
            intra_feat = F.interpolate(conv1, scale_factor=2, mode="bilinear") + self.conv_inner["2"](conv0)
            output_feature["2"] = self.conv_out["2"](intra_feat)
            
        return output_feature


class DeformStageNet(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None):
        super(DeformStageNet, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [32, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num
        # [B,8,H,W]
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        # [B,16,H/2,W/2]
        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        # [B,32,H/4,W/4]
        self.conv2 = nn.Sequential(
                Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = nn.Sequential(
            Conv2dUnit(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            )

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = DeformConv2dPack(in_channels=base_channels * 8, out_channels=self.output_channels[0], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["1"] = DeformConv2dPack(base_channels * 8, self.output_channels[1], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["2"] = DeformConv2dPack(base_channels * 4, self.output_channels[2], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["3"] = DeformConv2dPack(base_channels * 2, self.output_channels[3], kernel_size=3,stride=1,padding=1,deform_groups=1)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 8, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["3"] = nn.Conv2d(base_channels, base_channels * 2, 1, bias=True)
        

    def forward(self, x):
        output_feature = {}
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)
            output_feature["0"] = self.conv_out["0"](conv3)
            intra_feat = F.interpolate(conv3, scale_factor=2, mode="bilinear") + self.conv_inner["1"](conv2)
            output_feature["1"] = self.conv_out["1"](intra_feat)
            intra_feat = F.interpolate(conv2, scale_factor=2, mode="bilinear") + self.conv_inner["2"](conv1)
            output_feature["2"] = self.conv_out["2"](intra_feat)
            intra_feat = F.interpolate(conv1, scale_factor=2, mode="bilinear") + self.conv_inner["3"](conv0)
            output_feature["3"] = self.conv_out["3"](intra_feat)

        return output_feature

# not working better than StageFeatExtNet
class StageFeatExtNetFix(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None, interp_mode="bilinear"):
        super(StageFeatExtNetFix, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [64, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num
        self.interp_mode = interp_mode
        # [B,8,H,W]
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        # [B,16,H/2,W/2]
        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        # [B,32,H/4,W/4]
        self.conv2 = nn.Sequential(
                Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = nn.Sequential(
            Conv2dUnit(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            )

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = nn.Conv2d(base_channels * 8, self.output_channels[0], 1, bias=False)
            self.conv_out["1"] = nn.Conv2d(base_channels * 8, self.output_channels[1], 1, bias=False)
            self.conv_out["2"] = nn.Conv2d(base_channels * 8, self.output_channels[2], 1, bias=False)
            self.conv_out["3"] = nn.Conv2d(base_channels * 8, self.output_channels[3], 1, bias=False)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 8, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 8, 1, bias=True)
            self.conv_inner["3"] = nn.Conv2d(base_channels, base_channels * 8, 1, bias=True)
        
        if stage_num == 3:
            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = nn.Conv2d(base_channels * 4, self.output_channels[0], 1, bias=False)
            self.conv_out["1"] = nn.Conv2d(base_channels * 4, self.output_channels[1], 1, bias=False)
            self.conv_out["2"] = nn.Conv2d(base_channels * 4, self.output_channels[2], 1, bias=False)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["1"] = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels, base_channels * 4, 1, bias=True)

    def forward(self, x):
        output_feature = {}
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)
            output_feature["0"] = self.conv_out["0"](conv3)
            intra_feat = F.interpolate(conv3, scale_factor=2, mode=self.interp_mode) + self.conv_inner["1"](conv2)
            output_feature["1"] = self.conv_out["1"](intra_feat)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode=self.interp_mode) + self.conv_inner["2"](conv1)
            output_feature["2"] = self.conv_out["2"](intra_feat)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode=self.interp_mode) + self.conv_inner["3"](conv0)
            output_feature["3"] = self.conv_out["3"](intra_feat)

        if self.stage_num == 3:
            output_feature["0"] = self.conv_out["0"](conv2)
            intra_feat = F.interpolate(conv2, scale_factor=2, mode=self.interp_mode) + self.conv_inner["1"](conv1)
            output_feature["1"] = self.conv_out["1"](intra_feat)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode=self.interp_mode) + self.conv_inner["2"](conv0)
            output_feature["2"] = self.conv_out["2"](intra_feat)
            
        return output_feature

# not working better than DeformStageNet
class DeformStageNet1(nn.Module):
    def __init__(self, base_channels, stage_num=4, output_channels=None):
        super(DeformStageNet1, self).__init__()
        if output_channels is None:
            if stage_num == 4:
                self.output_channels = [64, 32, 16, 8]
            elif stage_num == 3:
                self.output_channels = [32, 16, 8]
        else:
            self.output_channels = output_channels
        self.base_channels = base_channels
        self.stage_num = stage_num
        # [B,8,H,W]
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        # [B,16,H/2,W/2]
        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        # [B,32,H/4,W/4]
        self.conv2 = nn.Sequential(
                Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        if stage_num == 4:
            # [B,64,H/8,W/8]
            self.conv3 = nn.Sequential(
            Conv2dUnit(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2dUnit(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            )

            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = DeformConv2dPack(in_channels=base_channels * 8, out_channels=self.output_channels[0], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["1"] = DeformConv2dPack(base_channels * 4, self.output_channels[1], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["2"] = DeformConv2dPack(base_channels * 2, self.output_channels[2], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["3"] = DeformConv2dPack(base_channels * 1, self.output_channels[3], kernel_size=3,stride=1,padding=1,deform_groups=1)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["0"] = nn.Conv2d(base_channels * 8, base_channels * 4, 1, bias=True)
            self.conv_inner["1"] = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=True)
            self.conv_inner["2"] = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=True)
        
        if stage_num == 3:
            self.conv_out = nn.ModuleDict()
            self.conv_out["0"] = DeformConv2dPack(in_channels=base_channels * 4, out_channels=self.output_channels[0], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["1"] = DeformConv2dPack(base_channels * 2, self.output_channels[1], kernel_size=3,stride=1,padding=1,deform_groups=1)
            self.conv_out["2"] = DeformConv2dPack(base_channels * 1, self.output_channels[2], kernel_size=3,stride=1,padding=1,deform_groups=1)

            self.conv_inner = nn.ModuleDict()
            self.conv_inner["0"] = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=True)
            self.conv_inner["1"] = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=True)

    def forward(self, x):
        output_feature = {}
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        if self.stage_num == 4:
            conv3 = self.conv3(conv2)

            output_feature["0"] = self.conv_out["0"](conv3)

            intra_feat = self.conv_inner["0"](conv3)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + conv2
            output_feature["1"] = self.conv_out["1"](intra_feat)

            intra_feat = self.conv_inner["1"](intra_feat)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + conv1
            output_feature["2"] = self.conv_out["2"](intra_feat)

            intra_feat = self.conv_inner["2"](intra_feat)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + conv0
            output_feature["3"] = self.conv_out["3"](intra_feat)

        if self.stage_num == 3:
            output_feature["0"] = self.conv_out["0"](conv2)

            intra_feat = self.conv_inner["0"](conv2)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + conv1
            output_feature["1"] = self.conv_out["1"](intra_feat)

            intra_feat = self.conv_inner["1"](intra_feat)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + conv0
            output_feature["2"] = self.conv_out["2"](intra_feat)
            
        return output_feature

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels=1):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, base_channels, padding=1, norm="groupnorm")

        self.conv1 = Conv3dUnit(base_channels, base_channels * 2, stride=(1, 2, 2), padding=1, norm="groupnorm")
        self.conv2 = Conv3dUnit(base_channels * 2, base_channels * 2, padding=1, norm="groupnorm")

        self.conv3 = Conv3dUnit(base_channels * 2, base_channels * 4, stride=(1, 2, 2), padding=1, norm="groupnorm")
        self.conv4 = Conv3dUnit(base_channels * 4, base_channels * 4, padding=1, norm="groupnorm")

        self.conv5 = Conv3dUnit(base_channels * 4, base_channels * 8, stride=(1, 2, 2), padding=1, norm="groupnorm")
        self.conv6 = Conv3dUnit(base_channels * 8, base_channels * 8, padding=1, norm="groupnorm")

        self.deconv7 = Deconv3dUnit(base_channels * 8, base_channels * 4, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="groupnorm")

        self.deconv8 = Deconv3dUnit(base_channels * 4, base_channels * 2, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="groupnorm")

        self.deconv9 = Deconv3dUnit(base_channels * 2, base_channels * 1, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="groupnorm")

        self.prob = nn.Conv3d(base_channels, out_channels, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x

class CostRegNetBN(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels=1):
        super(CostRegNetBN, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, base_channels, padding=1, norm="batchnorm")

        self.conv1 = Conv3dUnit(base_channels, base_channels * 2, stride=(1, 2, 2), padding=1, norm="batchnorm")
        self.conv2 = Conv3dUnit(base_channels * 2, base_channels * 2, padding=1, norm="batchnorm")

        self.conv3 = Conv3dUnit(base_channels * 2, base_channels * 4, stride=(1, 2, 2), padding=1, norm="batchnorm")
        self.conv4 = Conv3dUnit(base_channels * 4, base_channels * 4, padding=1, norm="batchnorm")

        self.conv5 = Conv3dUnit(base_channels * 4, base_channels * 8, stride=(1, 2, 2), padding=1, norm="batchnorm")
        self.conv6 = Conv3dUnit(base_channels * 8, base_channels * 8, padding=1, norm="batchnorm")

        self.deconv7 = Deconv3dUnit(base_channels * 8, base_channels * 4, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="batchnorm")

        self.deconv8 = Deconv3dUnit(base_channels * 4, base_channels * 2, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="batchnorm")

        self.deconv9 = Deconv3dUnit(base_channels * 2, base_channels * 1, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1), norm="batchnorm")

        self.prob = nn.Conv3d(base_channels, out_channels, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x

class FeatureFetcher(nn.Module):
    def __init__(self, mode="bilinear", align_corners=True):
        super(FeatureFetcher, self).__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """adapted from 
        https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/feature_fetcher.py

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            # hanlding z == 0.0
            # following https://github.com/kwea123/CasMVSNet_pl/blob/5813306b451b22226a321d347e5f6020a20ae8c9/models/modules.py#L75
            z_zero_mask = (z == 0.0)
            z[z_zero_mask] = 1.0

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            # hanlding z == 0.0
            uv[:, :, 0][z_zero_mask] = 2 * width
            uv[:, :, 1][z_zero_mask] = 2 * height

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
            grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

        pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode, align_corners=self.align_corners)
        pts_feature = pts_feature.squeeze(3)

        pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)

        return pts_feature


def get_pixel_grids(height, width):
    with torch.no_grad():
        # texture coordinate
        x_linspace = torch.linspace(0.5, width - 0.5, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0.5, height - 0.5, height).view(height, 1).expand(height, width)
        # y_coordinates, x_coordinates = torch.meshgrid(y_linspace, x_linspace)
        x_coordinates = x_linspace.contiguous().view(-1)
        y_coordinates = y_linspace.contiguous().view(-1)
        ones = torch.ones(height * width)
        indices_grid = torch.stack([x_coordinates, y_coordinates, ones], dim=0)
    return indices_grid


# estimate pixel-wise view weight
class PixelwiseNet(nn.Module):
    def __init__(self, in_channels):
        super(PixelwiseNet, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, 16, kernel_size=1, stride=1, padding=0, norm="batchnorm")
        self.conv1 = Conv3dUnit(16, 8, kernel_size=1, stride=1, padding=0, norm="batchnorm")
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
        

    def forward(self, x1):
        # x1: [B, G, Ndepth, H, W]
        
        # [B, Ndepth, H, W]
        x1 =self.conv2(self.conv1(self.conv0(x1))).squeeze(1)
        
        output = self.output(x1)
        del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]
        
        return output.unsqueeze(1) # [B,1,H,W]
