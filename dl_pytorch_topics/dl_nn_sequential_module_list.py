import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_cht = out_ch

        # Sequential connects modules in the order you add them. PyTorch tacks parameters of them
        self.main = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), # Preserve spatial dim.
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch)
                )


        # if out_ch 1= in_ch, use 1x1 conv to change.
        self.skip = (nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else
                     nn.Indentity())


    def forward(self, x: torch.tensor):
        assert 4 == x.dim()
        _, C, _, _ = x.shape

        assert C == self.in_ch

        return F.relu(self.main(x) + self.skip(x))


class ResNetStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks):
        '''
        num_blocks is a varialbe. Hence, we use nn.ModuleList
        '''
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock(in_ch, out_ch) if i == 0 else ResBlock(out_ch, out_ch) for i in range(num_blocks)]
        )

    def forward(self, x: torch.tensor):
        for block in self.blocks:
            x = block(x)

        return x


class ResNetStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks):
        super().__init__()
        # nn.ModuleLst allow dynamic models
        self.blocks = nn.ModuleList(
            [ResBlock(in_ch, out_ch if i == 0 else out_ch, out_ch) for i in range(num_blocks)]
        )

    def forward(self, x):
        '''
        Why not to use python list?
        layers = [nn.Linear(64, 64) for _ in range(5)]
        self.layers = layers  # ❌ Wrong! These won’t be registered as modules
        '''
        # self.blocks is nn.ModuleLst such that PyTorch tracks the parameters of them
        for block in self.blocks:
            x = block(x)
        return x

class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = ResNetStage(64, 64, num_blocks=2)
        self.stage2 = ResNetStage(64, 128, num_blocks=2)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.classifier(x)
