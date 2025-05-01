import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# Inicializace vah
def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# Dvojitá konvoluce
def double_conv(in_ch, mid_ch, out_ch):
    conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    return conv


# Oříznutí tensoru tak, aby odpovídal velikosti target_tensor (pro skip connections)
def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    
    if delta % 2 == 0:
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]
    else:
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta-1, delta:tensor_size-delta-1]
    
def crop_1(tensor):
    tensor_size = tensor.size()[2]
    if tensor_size % 2 != 0:
        return tensor[:, :, :+1, :+1]
    else:
        return tensor

def pad_tensor(tensor, pad_bottom=1, pad_right=1):
    tensor_size = tensor.size()[2]
    if tensor_size % 2 != 0:
        return F.pad(tensor, (0, pad_right, 0, pad_bottom))
    else:
        return tensor

'''
Třída pro implementaci architektury modelu UNet.
Inspirace z https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py
Vstupy:
    Počet výstupních kanálů - třídy, které rozlišuje
Výstup:
    Predikce pro vstupní obrázek
'''
class NestedUNet(nn.Module):
    def __init__(self, out_channels=5):
        super().__init__()

        n = 64
        filters = [n, n * 2, n * 4, n * 8, n * 16]

        # Max pooling mezi úrovněmi
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Down part (encoder)
        self.conv0_0 = double_conv(1, filters[0], filters[0])
        self.conv1_0 = double_conv(filters[0], filters[1], filters[1])
        self.conv2_0 = double_conv(filters[1], filters[2], filters[2])
        self.conv3_0 = double_conv(filters[2], filters[3], filters[3])
        self.conv4_0 = double_conv(filters[3], filters[4], filters[4])

        # Up part (decoder)
        self.conv0_1 = double_conv(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = double_conv(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = double_conv(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = double_conv(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = double_conv(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = double_conv(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = double_conv(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = double_conv(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = double_conv(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = double_conv(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)



    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.max_pool_2x2(x0_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.max_pool_2x2(x1_0))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.max_pool_2x2(x2_0))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))


        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.max_pool_2x2(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))


        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output
