from typing import List, Any

import torch
from torch import nn, Tensor



import torch.nn.functional as F
import math


XFORMERS_AVAILABLE = False



class MLPAdapter_Fixed(nn.Module):
    def __init__(self, fc_in_channels: int, in_channels: int, skip_connect=False):
        super().__init__()
        self.skip_connect = skip_connect
        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)
        # zero-init final linear to make adapter initially a no-op
        nn.init.zeros_(self.D_fc2.weight)
        if self.D_fc2.bias is not None:
            nn.init.zeros_(self.D_fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.D_fc1(x)
        x0 = F.gelu(x0)   # match main net activation
        outputs = self.D_fc2(x0)
        if self.skip_connect:
            outputs = outputs + x
        return outputs






class MulConvAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,#1024
        in_channels: int,#512
        ch1x1: int,#256
        ch3x3red: int,#32
        ch3x3: int,#128
        ch5x5red: int,#32
        ch5x5: int,#128
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect
        conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.D_fc1(x)
        B,P,D = x0.shape


        W = int ( (math.sqrt((P-5)/12))*4 )#5 special token are added
        H = int ( W / 4 * 3 )
        #W = H = int(math.sqrt(P-1))

        x0 = F.relu(x0, inplace=True)
        

        xs = x0[:,5:,:]
        xs = xs.reshape(B,W,H,D).permute(0,3,1,2)
        branch1 = self.branch1(xs)
        branch2 = self.branch2(xs)
        branch3 = self.branch3(xs)
        outputs = [branch1, branch2, branch3]
        outputs = torch.cat(outputs,dim=1)
        outputs = outputs.reshape(B,D,W*H).permute(0,2,1)
        clstoken =  x0[:,0:5,:]
        outputs = torch.cat([clstoken,outputs],dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)
        if self.skip_connect:
            outputs+=x
        return outputs

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    






class MLPAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int, #1024
        in_channels: int, #512
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect

        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.D_fc1(x)

        x0 = F.relu(x0, inplace=True)
        
        outputs = self.D_fc2(x0)

        if self.skip_connect:
            outputs+=x
        return outputs
    



class DwAdapter(nn.Module):
    def __init__(self, C, kt=3, K=5):
        super().__init__()
        self.K = K
        self.ln = nn.LayerNorm(C)
        self.dw_t = nn.Conv1d(C, C, kt, padding=kt//2, groups=C, bias=False)
        self.pw = nn.Conv1d(C, C, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, H=15, W=20, S=9, P=305):
        B, SP, C = x.shape
        x0 = x.reshape(B, S, P, C)

        # LN on feature dim
        x0 = self.ln(x0)

        # process tokens after K
        x1 = x0[:, :, self.K:, :]                  # [B,S,P-K,C]
        outputs = x1.permute(0,2,3,1).reshape(B*(P-self.K), C, S)  # [B(P-K),C,S]

        outputs = self.pw(self.act(self.dw_t(outputs)))
        outputs = outputs.reshape(B, P-self.K, C, S).permute(0,3,1,2)  # [B,S,P-K,C]

        # concat preserved + adapted
        outputs = torch.cat([x0[:, :, :self.K, :], outputs], dim=2)    # [B,S,P,C]

        outputs = outputs.reshape(B, SP, C)
        return x + outputs

        