import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


##########################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

##########################################################################


class RSAB(nn.Module):
    def __init__(self, n_feat, kernel_size):

        super(RSAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.sa = spatial_attn_layer()


    def forward(self, x):
        res = self.body(x)
        res = self.sa(res)
        res = res + x
        return res


## Residual Group (RG)
class spaResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(spaResidualGroup, self).__init__()

        self.body1 = RSAB(n_feat, kernel_size)
        self.body2 = RSAB(n_feat, kernel_size)
        self.body3 = RSAB(n_feat, kernel_size)
        self.body4 = RSAB(n_feat, kernel_size)
        self.body5 = RSAB(n_feat, kernel_size)
        self.conv0 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)


    def forward(self, x):
        res = self.body1(x)
        res = self.body2(res)
        res = self.body3(res)
        res = self.body4(res)
        res = self.body5(res)
        res = self.conv0(res)
        res += x
        return res



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size):

        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.ca = CALayer(n_feat)


    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res = res + x
        return res

## Residual Group (RG)
class speResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(speResidualGroup, self).__init__()

        self.body1 = RCAB(n_feat, kernel_size)
        self.body2 = RCAB(n_feat, kernel_size)
        self.body3 = RCAB(n_feat, kernel_size)
        self.body4 = RCAB(n_feat, kernel_size)
        self.body5 = RCAB(n_feat, kernel_size)
        self.conv0 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)


    def forward(self, x):
        res = self.body1(x)
        res = self.body2(res)
        res = self.body3(res)
        res = self.body4(res)
        res = self.body5(res)
        res = self.conv0(res)
        res += x
        return res


##########################################################################


## Dual Attention Block (DAB)
class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        )

        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


## Recursive Residual Group (RRG)
class RRG(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction=16, act=nn.ReLU(True),  num_dab = 5):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [
            DAB(
                n_feat, kernel_size, reduction, bias=True, bn=False, act=act) \
            for _ in range(num_dab)]
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.conv(res)
        res += x
        return res


##########################################################################


class SpatialBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(SpatialBlock, self).__init__()
        self.spatial_layers = spaResidualGroup(n_channels, kernel_size=kernel_size)

    def forward(self, spatial_x):
        spatial_x = self.spatial_layers(spatial_x)
        return spatial_x


class SpectralBlock(nn.Module):
    def __init__(self,kernel_size=3, n_channels=64):
        super(SpectralBlock, self).__init__()
        self.spectral_layers = speResidualGroup(n_channels, kernel_size=kernel_size)

    def forward(self, spectral_x):
        spectral_x = self.spectral_layers(spectral_x)
        return spectral_x




class FusionBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(FusionBlock, self).__init__()
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(n_channels *2, n_channels *2, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels * 2, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.spatial_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.spectral_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spatial_x, spectral_x):
        ss_x = torch.cat([spatial_x, spectral_x], dim=1)
        ss_x = self.fusion_layers(ss_x)
        spatial_x = spatial_x + self.spatial_broadcast(ss_x)
        spatial_x = self.relu(spatial_x)
        spectral_x = spectral_x + self.spectral_broadcast(ss_x)
        spectral_x = self.relu(spectral_x)

        return ss_x, spatial_x, spectral_x


class FusionBlock2(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(FusionBlock2, self).__init__()
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(n_channels *3, n_channels *3, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels * 3, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.spatial_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.spectral_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ss_x, spatial_x, spectral_x):
        ss_x2 = torch.cat([ss_x, spatial_x, spectral_x], dim=1)
        ss_x2 = self.fusion_layers(ss_x2)
        spatial_x = spatial_x + self.spatial_broadcast(ss_x2)
        spatial_x = self.relu(spatial_x)
        spectral_x = spectral_x + self.spectral_broadcast(ss_x2)
        spectral_x = self.relu(spectral_x)

        return ss_x2, spatial_x, spectral_x


class FusionBlock3(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(FusionBlock3, self).__init__()
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(n_channels *4, n_channels *4, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels * 4, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.spatial_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.spectral_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ss_x, ss_x2, spatial_x, spectral_x):
        ss_x3 = torch.cat([ss_x, ss_x2, spatial_x, spectral_x], dim=1)
        ss_x3 = self.fusion_layers(ss_x3)
        spatial_x = spatial_x + self.spatial_broadcast(ss_x3)
        spatial_x = self.relu(spatial_x)
        spectral_x = spectral_x + self.spectral_broadcast(ss_x3)
        spectral_x = self.relu(spectral_x)

        return ss_x3, spatial_x, spectral_x




class SpatialSpectralSRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_channels=64, n_blocks=7, kernel_size=3, upscale_factor=2):
        super(SpatialSpectralSRNet, self).__init__()
        self.n_blocks = n_blocks
        self.pre_spatial_layers = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True)
        )
        self.pre_spectral_layers = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True)
        )
        relu = nn.ReLU(inplace=True)

        self.spa1 = SpatialBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.spe1 = SpectralBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.fusion1 = FusionBlock(kernel_size = kernel_size, n_channels=n_channels)

        self.spa2 = SpatialBlock(kernel_size = kernel_size, n_channels=n_channels)
        self.spe2 = SpectralBlock(kernel_size = kernel_size, n_channels=n_channels)
        self.fusion2 = FusionBlock2(kernel_size = kernel_size, n_channels=n_channels)

        self.spa3 = SpatialBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.spe3 = SpectralBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.fusion3 = FusionBlock3(kernel_size = kernel_size, n_channels=n_channels)

        # isolated spatial and spectral loss layers
        self.post_spatial_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(in_channels=n_channels//4, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.post_spectral_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
        )

        self.fusion_net =  nn.Sequential(
            # RRG(n_channels, kernel_size=kernel_size),
            # RRG(n_channels, kernel_size=kernel_size),
            # RRG(n_channels, kernel_size=kernel_size),
            RRG(n_channels, kernel_size=kernel_size)
        )


        self.conv1 = nn.Conv2d(in_channels=4*n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.pre_fusion_layers = nn.Sequential(
            nn.Conv2d(in_channels=5 * n_channels, out_channels=4 * n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )


        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )

    def forward(self, x):
        spatial_x = self.pre_spatial_layers(x)
        spectral_x = self.pre_spectral_layers(x)

        spatial_x_res = self.spa1(spatial_x)
        spectral_x_res = self.spe1(spectral_x)
        ss_x, spatial_x_res, spectral_x_res = self.fusion1(spatial_x_res, spectral_x_res)

        spatial_x_res = self.spa2(spatial_x_res)
        spectral_x_res = self.spe2(spectral_x_res)
        ss_x2, spatial_x_res, spectral_x_res = self.fusion2(ss_x, spatial_x_res, spectral_x_res)

        spatial_x_res = self.spa3(spatial_x_res)
        spectral_x_res = self.spe3(spectral_x_res)
        ss_x3, spatial_x_res, spectral_x_res = self.fusion3(ss_x, ss_x2, spatial_x_res, spectral_x_res)


        spatial_x = spatial_x + spatial_x_res
        spectral_x = spectral_x + spectral_x_res
        out_spatial = self.post_spatial_layers(spatial_x)
        out_spectral = self.post_spectral_layers(spectral_x)

        x = torch.cat([ss_x, ss_x2, ss_x3, spatial_x, spectral_x], dim=1)
        x = self.pre_fusion_layers(x)
        res1 = self.fusion_net(x)
        res2 = self.fusion_net(res1)
        res3 = self.fusion_net(res2)
        res4 = self.fusion_net(res3)
        res = torch.cat([res1, res2, res3, res4], dim=1)
        res = self.conv1(res)
        x = x + res
        out = self.fusion_block(x)
        return out_spatial, out_spectral, out
