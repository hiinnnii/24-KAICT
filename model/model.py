import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *
import torch.utils.checkpoint as checkpoint
# UNet with Xception blocks in the encoder
class UNet_Xception(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Xception, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Xception blocks used for the encoder (downsampling)
        self.inc = Block(n_channels, 64, reps=2, stride=1, start_with_relu=False, grow_first=True)
        self.down1 = Block(64, 128, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.down2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.down3 = Block(256, 512, reps=2, stride=2, start_with_relu=True, grow_first=True)
        factor = 2 if bilinear else 1
        self.down4 = Block(512, 1024 // factor, reps=2, stride=2, start_with_relu=True, grow_first=False)
        
        # Regular UNet upsampling (decoder)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
'''
class UNetWithCheckpoint(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetWithCheckpoint, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = nn.Dropout(p=0.2)

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = checkpoint.checkpoint(self.inc,x)
        x1 = checkpoint.checkpoint(self.dropout,x1)  # 드롭아웃 추가
        x2 = checkpoint.checkpoint(self.down1,x1)
        x2 = checkpoint.checkpoint(self.dropout,x2)
        x3 = checkpoint.checkpoint(self.down2,x2)
        x3 = checkpoint.checkpoint(self.dropout,x3)
        x4 = checkpoint.checkpoint(self.down3,x3)
        x4 = checkpoint.checkpoint(self.dropout,x4)
        x5 = checkpoint.checkpoint(self.down4,x4)
        x = checkpoint.checkpoint(self.up1,x5, x4)
        x = checkpoint.checkpoint(self.up2,x, x3)
        x = checkpoint.checkpoint(self.up3,x, x2)
        x = checkpoint.checkpoint(self.up4,x, x1)
        logits = self.outc(x)
        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = nn.Dropout(p=0.2)

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = checkpoint.checkpoint(self.inc(x))
        x1 = self.dropout(x1)  # 드롭아웃 추가
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()


        self.enc_conv0 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upsample0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv0 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):

        x = self.enc_conv0(x)
        x = torch.relu(x)
        x = self.pool0(x)
        
        x = self.enc_conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.upsample0(x)
        x = self.dec_conv0(x)
        x = torch.relu(x)
        x = self.upsample1(x)
        x = self.dec_conv1(x)
        x = torch.relu(x)
        
        x = self.final_conv(x)
        return torch.sigmoid(x)
        
        
        
'''