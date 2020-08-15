import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NOTE: the 3 customisable U-Net classes below are adapted by the creators of RRIN from https://github.com/jvanvugt/pytorch-unet
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, filter_num=5, padding=True,):
        super(UNet, self).__init__()

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (filter_num + i), padding)
            )
            prev_channels = 2 ** (filter_num + i)
        self.midconv = nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (filter_num + i), padding)
            )
            prev_channels = 2 ** (filter_num + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.midconv(x), negative_slope = 0.1)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetUpBlock, self).__init__()

        self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            )
        self.conv_block = UNetConvBlock(in_size, out_size, padding)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
    
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]
    
    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat((up, crop1), 1)
        out = self.conv_block(out)
    
        return out


# NOTE: this is the original RRIN model we built our adapted model upon
# def warp(img, flow):
#     _, _, H, W = img.size()
#     gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
#     gridX = torch.tensor(gridX, requires_grad=False)
#     gridY = torch.tensor(gridY, requires_grad=False)
#     u = flow[:,0,:,:]
#     v = flow[:,1,:,:]
#     x = gridX.unsqueeze(0).expand_as(u).float()+u
#     y = gridY.unsqueeze(0).expand_as(v).float()+v
#     normx = 2*(x/W-0.5)
#     normy = 2*(y/H-0.5)
#     grid = torch.stack((normx,normy), dim=3)
#     warped = F.grid_sample(img, grid)
#     return warped


# class RRIN(nn.Module):
#     def __init__(self,level=3):
#         super(RRIN, self).__init__()
#         self.Mask = UNet(16,2,4)
#         self.Flow_L = UNet(6,4,5)
#         self.refine_flow = UNet(10,4,4)
#         self.final = UNet(9,3,4)

#     def process(self,x0,x1,t):

#         x = torch.cat((x0,x1),1)
#         Flow = self.Flow_L(x)
#         Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
#         Flow_t_0 = -(1-t)*t*Flow_0_1+t*t*Flow_1_0
#         Flow_t_1 = (1-t)*(1-t)*Flow_0_1-t*(1-t)*Flow_1_0
#         Flow_t = torch.cat((Flow_t_0,Flow_t_1,x),1)
#         Flow_t = self.refine_flow(Flow_t)
#         Flow_t_0 = Flow_t_0+Flow_t[:,:2,:,:]
#         Flow_t_1 = Flow_t_1+Flow_t[:,2:4,:,:]
#         xt1 = warp(x0,Flow_t_0)
#         xt2 = warp(x1,Flow_t_1)
#         temp = torch.cat((Flow_t_0,Flow_t_1,x,xt1,xt2),1)
#         Mask = torch.sigmoid(self.Mask(temp))
#         w1, w2 = (1-t)*Mask[:,0:1,:,:], t*Mask[:,1:2,:,:]
#         output = (w1*xt1+w2*xt2)/(w1+w2+1e-8)

#         return output

#     def forward(self, input0, input1, t=0.5):

#         output = self.process(input0,input1,t)
#         compose = torch.cat((input0, input1, output),1)
#         final = self.final(compose)+output
#         final = final.clamp(0,1)

#         return final


# NOTE: we mark our changes to the original RRIN model by commenting out the parts that we removed, i.e. UNet_2 and UNet_4
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # this U-Net produces the coarse bidirectional optical flow estimates
        self.first_flow = UNet(6,4,5)
        # self.refine_flow = UNet(10,4,4)
        # this U-Net produces the weight maps
        self.weight_map = UNet(16,2,4)
        # self.final = UNet(9,3,4)

    def warp(self, img, flow):
        """
        Warps input image tensors with corresponding optical flows.

        Args:
            img (Tensor): Input image tensors.
            flow (Tensor): Input optical flows.

        Returns:
            warped (Tensor): Image tensors warped with optical flows.
        """
        _, _, H, W = img.size()
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        gridX = torch.tensor(gridX, requires_grad=False).to(device)
        gridY = torch.tensor(gridY, requires_grad=False).to(device)
        u = flow[:,0,:,:]
        v = flow[:,1,:,:]
        x = gridX.unsqueeze(0).expand_as(u).float() + u
        y = gridY.unsqueeze(0).expand_as(v).float() + v
        normx = 2*(x / W - 0.5)
        normy = 2*(y / H - 0.5)
        grid = torch.stack((normx, normy), dim=3)
        warped = F.grid_sample(img, grid, align_corners=True)

        return warped

    def process(self, frame0, frame1, t):
        """
        Main part of forward pass of model.

        Args:
            frame0 (Tensor): First frames' image tensors.
            frame1 (Tensor): Last frames' image tensors.
            t (float, optional): Time interval between frame0 and frame1 to generate the interpolated frame for, ranges from 0 to 1.

        Returns:
            output (Tensor): Interpolated frames' image tensors warped with optical flows and processed with weight maps.
            flow_t_0 (Tensor): Optical flow estimate for t and 0.
            flow_t_1 (Tensor): Optical flow estimate for t and 1.
            w1 (Tensor): Weight map for t and 0.
            w2 (Tensor): Weight map for t and 1.
        """
        # get bidrectional flow
        x = torch.cat((frame0, frame1), 1)
        flow = self.first_flow(x)
        flow_0_1, flow_1_0 = flow[:,:2,:,:], flow[:,2:4,:,:]
        flow_t_0 = -(1-t) * t * flow_0_1 + t * t * flow_1_0
        flow_t_1 = (1-t) * (1-t) * flow_0_1 - t * (1-t) * flow_1_0

        # refine flow
        # flow_t = torch.cat((flow_t_0, flow_t_1, x), 1)
        # flow_t = self.refine_flow(flow_t)
        # flow_t_0 = flow_t_0 + flow_t[:,:2,:,:]
        # flow_t_1 = flow_t_1 + flow_t[:,2:4,:,:]

        # warping
        xt1 = self.warp(frame0, flow_t_0)
        xt2 = self.warp(frame1, flow_t_1)

        # get weight map
        temp = torch.cat((flow_t_0, flow_t_1, x, xt1, xt2), 1)
        mask = torch.sigmoid(self.weight_map(temp))
        w1, w2 = (1-t) * mask[:,0:1,:,:], t * mask[:,1:2,:,:]
        
        # get final coarse output
        output = (w1 * xt1 + w2 * xt2) / (w1 + w2 + 1e-8)

        return output, flow_t_0, flow_t_1, w1, w2
    
    def forward(self, frame0, frame1, t=0.5):
        """
        Forward pass of model.

        Args:
            frame0 (Tensor): First frames' image tensors.
            frame1 (Tensor): Last frames' image tensors.
            t (float, optional): Time interval between frame0 and frame1 to generate the interpolated frame for, ranges from 0 to 1.
        """
        output, flow_t_0, flow_t_1, w1, w2 = self.process(frame0, frame1, t)
        # compose = torch.cat((frame0, frame1, output), 1)
        # final = self.final(compose) + output
        # final = final.clamp(0,1)

        # make sure final output values are between 0 and 1, i.e. valid image tensors 
        final = output.clamp(0,1)

        return final, flow_t_0, flow_t_1, w1, w2


def normal_init(m, mean, std):
    """
    Instantiates weights for specific layer types in PyTorch with values drawn from a normal distribution.
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self):
        """
        PatchGAN discriminator.
        """
        super(Discriminator, self).__init__()

        c = 64
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=c*8, out_channels=1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )


    def weight_init(self, mean, std):
        """
        Allows for instantiation of discriminator weights as values drawn from a normal distribution.

        Args:
            mean (float): Mean of normal distribution.
            std (float): Standard deviation of normal distribution.
        """
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def forward(self, first, mid, last):
        """
        Forward pass of PatchGAN.

        Args:
            first (Tensor): First frames' image tensors.
            mid (Tensor): Middle frames' image tensors, can be real or generated.
            last (Tensor): Last frames' image tensors.

        Returns:
            x (Tensor): Patches with values between 0 and 1 due to the sigmoid activation.
        """
        x = torch.cat([first, mid, last], dim=1)
        x = self.model(x)

        return x 