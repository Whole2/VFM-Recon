import torch
import torch.nn as nn
import torchvision
from vggt.models.aggregator import Aggregator
from vggt.heads.dpt_head import DPTHead
from vggt.models.vggt import VGGT #xmodified

class MVViTEncoder(torch.nn.Module):
    def __init__(
            self, 
            img_size=518, 
            patch_size=14, 
            embed_dim=1024,
            output_dim=64,
            vit_size=(168, 224),
            ckpt=None):
        super().__init__()

        assert ckpt is not None, "Please provide a checkpoint for the ViT encoder"
        
        self.vggt=VGGT(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim) #xmodified
        state_dict = torch.hub.load_state_dict_from_url(
            "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
        ) #xmodified url
        self.vggt.load_state_dict(state_dict, strict=False)# xmodified: strict=false

        self.vit_size = vit_size
        # self.aggregator = Aggregator(
        #     img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        # self.aggregator.load_state_dict(torch.load(ckpt, weights_only=False))
        self.aggregator = self.vggt.aggregator  #xmodified
        
        head_features = 256
        head_features_2 = 128
        self.dpt_head = DPTHead(
            dim_in=2 * embed_dim, 
            output_dim=output_dim,
            features=head_features,
            feature_only=True)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(head_features, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )
        
        #  freeze backbone, train adapter
        for name, param in self.aggregator.named_parameters():
            if "adapter" in name:
                param.requires_grad = True   # train adapter
            else:
                param.requires_grad = False  # freeze backbone
        
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        B, S, _, H, W = x.shape
        x = x.reshape(B*S, -1, H, W)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
              x = torch.nn.functional.interpolate(
                  x, size=self.vit_size, mode="bilinear", align_corners=False)
        
        x = x.reshape(B, S, -1, *self.vit_size)
        aggregated_tokens_list, patch_start_idx = self.aggregator(x)
        feat = self.dpt_head(aggregated_tokens_list, images=x, patch_start_idx=patch_start_idx)
        feat = self.output_conv(feat.reshape(B*S, -1, *feat.shape[-2:]))
        return feat
    
class Cnn3d(torch.nn.Module):
    def __init__(self, in_c):
        super().__init__()

        channels = [64, 64, 128, 64, 64]

        self.stem = torch.nn.Sequential(
            ConvBnRelu3d(in_c, channels[0], ks=1, padding=0),
            ResBlock3d(channels[0]),
        )
        self.conv1x1_1 = ConvBnRelu3d(channels[0], channels[1])
        self.down1 = torch.nn.Sequential(
            ResBlock3d(channels[1]),
            ResBlock3d(channels[1]),
        )
        self.conv1x1_2 = ConvBnRelu3d(channels[1], channels[2])
        self.down2 = torch.nn.Sequential(
            ResBlock3d(channels[2]),
            ResBlock3d(channels[2]),
        )
        self.up1 = torch.nn.Sequential(
            ConvBnRelu3d(channels[2] + channels[1], channels[3]),
            ResBlock3d(channels[3]),
            ResBlock3d(channels[3]),
        )
        self.up2 = torch.nn.Sequential(
            ConvBnRelu3d(channels[3] + channels[0], channels[4]),
            ResBlock3d(channels[4]),
            ResBlock3d(channels[4]),
        )
        self.up3 = torch.nn.Sequential(
            ConvBnRelu3d(channels[4] + in_c, channels[4]),
            ResBlock3d(channels[4]),
            ResBlock3d(channels[4]),
        )
        self.out_c = channels[4]

    def forward(self, x, _):
        x0 = self.stem(x)
        x1 = torch.nn.functional.max_pool3d(self.conv1x1_1(x0), 2)
        x1 = self.down1(x1)
        out = torch.nn.functional.max_pool3d(self.conv1x1_2(x1), 2)
        out = self.down2(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        out = torch.cat((out, x1), dim=1)
        out = self.up1(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        out = torch.cat((out, x0), dim=1)
        out = self.up2(out)
        out = torch.cat((out, x), dim=1)
        out = self.up3(out)
        return out


class Cnn2d(torch.nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        channel_mean = [0.485, 0.456, 0.406]
        channel_std = [0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(channel_mean, channel_std)

        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        backbone = torchvision.models.efficientnet_v2_s(weights=weights, progress=True)

        self.conv0 = backbone.features[:3]
        self.conv1 = backbone.features[3]
        self.conv2 = backbone.features[4]

        self.out0 = torch.nn.Sequential(
            torch.nn.Conv2d(48, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out3 = ResBlock2d(out_dim)

    def forward(self, x):
        x = self.normalize(x)

        x = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        x = self.out0(x)
        conv1 = self.out1(conv1)
        conv2 = self.out2(conv2)

        conv1 = torch.nn.functional.interpolate(
            conv1, scale_factor=2, mode="bilinear", align_corners=False
        )
        conv2 = torch.nn.functional.interpolate(
            conv2, scale_factor=4, mode="bilinear", align_corners=False
        )

        x += conv1
        x += conv2

        return self.out3(x)


class FeatureFusion(torch.nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.out_c = in_c
        self.bn = torch.nn.BatchNorm3d(self.out_c)

    def forward(self, x, valid):
        counts = torch.sum(valid, dim=1, keepdim=True)
        counts.masked_fill_(counts == 0, 1)
        x.masked_fill_(~valid[:, :, None], 0)
        x /= counts[:, :, None]
        mean = x.sum(dim=1)

        return self.bn(mean)


class ResBlock(torch.nn.Module):
    def forward(self, x):
        out = self.net(x)
        out += x
        torch.nn.functional.leaky_relu_(out)
        return out


class ResBlock3d(ResBlock):
    def __init__(self, c, ks=3, padding=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(c, c, ks, bias=False, padding=padding),
            torch.nn.BatchNorm3d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv3d(c, c, ks, bias=False, padding=padding),
            torch.nn.BatchNorm3d(c),
        )


class ResBlock2d(ResBlock):
    def __init__(self, c, ksize=3, padding=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(c, c, ksize, bias=False, padding=padding),
            torch.nn.BatchNorm2d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv2d(c, c, ksize, bias=False, padding=padding),
            torch.nn.BatchNorm2d(c),
        )


class ResBlock1d(ResBlock):
    def __init__(self, c):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(c, c, 1, bias=False),
            torch.nn.BatchNorm1d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv1d(c, c, 1, bias=False),
            torch.nn.BatchNorm1d(c),
        )


class ConvBnRelu3d(torch.nn.Module):
    def __init__(self, in_c, out_c, ks=3, padding=1):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_c, out_c, ks, padding=padding, bias=False),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.LeakyReLU(True),
        )

    def forward(self, x):
        return self.net(x)



if __name__ == "__main__":
    model = MVViTEncoder(output_dim=47,ckpt="111").cuda()
    # x = torch.rand(2, 2, 3, 480, 640).cuda()
    # out1 = model(x)

    # cnn = Cnn2d(64).cuda()
    # x = x.reshape(2*2, 3, 480, 640)
    # out2 = cnn(x)

    # out = torch.nn.functional.interpolate(out1.view(2*2, -1, *out1.shape[-2:]), size=out2.shape[-2:])
    # print(out1.shape, out2.shape, out.shape)
    
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n================= Parameter Count =================")

    # 1) Adapter 模块
    adapter_params = 0
    for name, param in model.aggregator.named_parameters():
        if "adapter" in name:
            adapter_params += param.numel()
    print(f"Adapter params: {adapter_params:,}")

    # 2) DPTHead
    print(f"DPTHead params: {count_parameters(model.dpt_head):,}")

    # 3) CNN3D
    cnn3d = Cnn3d(in_c=48).cuda()
    print(f"Cnn3d params: {count_parameters(cnn3d):,}")

    # 4) CNN2D
    cnn2d = Cnn2d(out_dim=47).cuda()
    print(f"Cnn2d params: {count_parameters(cnn2d):,}")

    print("===================================================\n")

