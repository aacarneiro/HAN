from torch import nn
import torch
from torchsummary import summary
from torch.autograd import Variable
import config


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.05, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, layer_channels, reduction=16):
        super().__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        reduced_channels = layer_channels//reduction
        self.conv_down = nn.Conv2d(
            layer_channels, reduced_channels, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU()
        self.conv_up = nn.Conv2d(
            reduced_channels, layer_channels, kernel_size=1, padding=0)
        self.sig = nn.Sigmoid()
        self.channel_attention = nn.Sequential(self.global_pooling,
                                               self.conv_down,
                                               self.relu,
                                               self.conv_up,
                                               self.sig)

    def forward(self, x):
        channels = self.channel_attention(x)
        out = torch.mul(channels, x)
        # print("Channel Attention Layer: ")
        # print(f"Channels: {channels.shape}. x: {x.shape}")
        # print(f"Out: {out.shape}")
        return out


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, layer_channels):
        super().__init__()
        self.conv_relu = ConvRelu(layer_channels, layer_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.channel_attention = ChannelAttention(
            layer_channels=layer_channels)

    def forward(self, x):
        x1 = self.conv_relu(x)
        ca = self.channel_attention(x1)
        out = x + ca
        return out


class ResidualGroup(nn.Module):
    def __init__(self, n_rcabs, layer_channels):
        super().__init__()
        self.rcabs = nn.ModuleList([ResidualChannelAttentionBlock(
            layer_channels) for _ in range(n_rcabs)])
        self.net = nn.Sequential(*self.rcabs)
        self.conv = nn.Conv2d(layer_channels, layer_channels,
                              kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        initial = x
        out = self.net(x)
        out = self.conv(out)
        # Long skip connection
        return out + initial


class ChannelSpatialAttention(nn.Module):
    def __init__(self, layer_channels):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=1, out_channels=1, kernel_size=(3, 3, 3), stride=1, padding='same')
        self.sig = nn.Sigmoid()

        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        initial = x
        x = x.unsqueeze(1)
        # print(f"CSAM internal shape - x: {x.shape}")
        # print(f"CSAM internal shape - initial: {initial.shape}")
        x = self.conv3d(x)
        # print(f"CSAM internal shape - after conv3d: {x.shape}")
        x = self.sig(x)
        x = x.squeeze(1)
        # print(f"CSAM internal shape - before mult: {x.shape}")
        # Element-wise product
        x = torch.mul(x, initial)
        out = self.beta*x + initial
        # print(f"CSAM internal shape - output: {out.shape}")
        return out


class LayerAttention(nn.Module):
    def __init__(self, num_resgroups, height, width, channels):
        super().__init__()
        self.N = num_resgroups
        self.height = height
        self.width = width
        self.channels = channels
        self.soft = nn.Softmax(dim=2)
        self.conv2d = nn.Conv2d(self.N*self.channels, out_channels=self.channels,
                                kernel_size=3, stride=1, padding=1)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        reshaped = torch.flatten(x, start_dim=2, end_dim=4)
        # print(f"LAM flat shape: {reshaped.shape}")
        # reshaped = torch.reshape(
        #     x, (self.N, self.height*self.width*self.channels))
        # print(f"LAM internal shape - reshaped input: {reshaped.shape}")
        transposed = torch.transpose(reshaped, 1, 2)
        # print(f"LAM internal shape - transposed input: {transposed.shape}")
        correlation = torch.matmul(reshaped, transposed)
        #   print(f"LAM internal shape - x1 input: {correlation.shape}")
        correlation = self.soft(correlation)
        out = torch.matmul(correlation, reshaped)
        out = torch.reshape(
            out, (batch_size, self.N, self.channels, self.height, self.width))
        # print(f"LAM output shape: {out.shape}")
        # print(f"LAM input shape: {x.shape}")
        out = self.alpha*out + x
        out = torch.reshape(
            out, (batch_size, self.N*self.channels, self.height, self.width))
        out = self.conv2d(out)
        # out = out.squeeze()
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        # in_c-4, H, W ->> in_c, H*2, W*2
        self.conv = nn.Conv2d(in_channels, in_channels*scale_factor**2,
                              kernel_size=3, stride=1, padding=1, bias=True)
        self.ps = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        return self.ps(self.conv(x))


class HAN(nn.Module):
    def __init__(self, num_resgroups, num_rcab, height=48, width=48, in_channels=3, layer_channels=64, kernel_conv=3, scale_factor=2):
        super().__init__()
        # The model is only valid for scale factors 2 and 3.
        assert(scale_factor == 2 or scale_factor == 3)

        self.num_resgroups = num_resgroups
        self.conv_initial = nn.Conv2d(in_channels=in_channels, out_channels=layer_channels,
                                      kernel_size=kernel_conv, stride=1, padding="same", bias=True)
        self.bn_HAN = nn.BatchNorm2d(layer_channels)
        self.RGs = nn.ModuleList([ResidualGroup(n_rcabs=num_rcab,
                                                layer_channels=layer_channels) for _ in range(num_resgroups)])
        self.channel_spatial_attention = ChannelSpatialAttention(
            layer_channels=layer_channels)
        self.layer_attention = LayerAttention(
            num_resgroups, height, width, layer_channels)
        self.upsample = UpsampleBlock(layer_channels, scale_factor)
        # self.feature_group = [torch.empty(
        #     (height, width, layer_channels)) for _ in range(num_resgroups)]
        self.conv_preCSAM = nn.Conv2d(in_channels=layer_channels, out_channels=layer_channels,
                                      kernel_size=3, stride=1, padding=1)
        self.conv_final = nn.Conv2d(in_channels=layer_channels, out_channels=in_channels,
                                    kernel_size=kernel_conv, stride=1, padding="same")

    def forward(self, x):
        # print(f"Input: {x.shape}")
        x = self.conv_initial(x)
        x = self.bn_HAN(x)
        # print(f"First Conv: {x.shape}")
        rg_input = x
        # Store the output of each resgroup to be used as
        # input to the LayerAttention module. 
        feature_group = []
        for n in range(self.num_resgroups):
            rg_input = self.RGs[n](rg_input)
            feature_group.append(rg_input)
        # print(f"Last RG Group: {feature_group[-1].shape}")
        # The last feature group is used as input to the Channel Attention
        # # after a convolution layer
        channel_attention_input = self.conv_preCSAM(feature_group[-1])

        # print(f"CSAM: {channel_attention_input.shape}")
        csam = self.channel_spatial_attention(channel_attention_input)

        # The input to the Layer Attention Module are the N feature groups stacked
        layer_attention_input = torch.stack(
            feature_group).permute(1, 0, 2, 3, 4)
        lam = self.layer_attention(layer_attention_input)
        # print(f"LAM: {layer_attention_input.shape}")
        out = csam + lam + x

        # out = channel_attention_input + x
        out = self.upsample(out)
        out = self.conv_final(out)
        return out


def test():
    low_resolution = 48  # 96x96 -> 48x48 2x Upscaling
    x = torch.randn((5, 3, low_resolution, low_resolution))
    y = torch.randn((5, 16, low_resolution, low_resolution))
    model = HAN(num_resgroups=config.N_GROUPS, num_rcab=config.N_RCAB, height=config.LOW_RES,
                width=config.LOW_RES, in_channels=3, layer_channels=config.N_CHANNELS)
    output = model(x)
    print(f"Output shape: {output.shape}")
    summary(model, (3, 48, 48), device="cpu")


if __name__ == "__main__":
    from torchsummary import summary
    test()
