import torch
import torch.nn as nn
import torch.nn.init as init

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
def double_conv(in_ch, out_ch):
    conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    return conv

# Oříznutí tensoru tak, aby odpovídal velikosti target_tensor (pro skip connections)
def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


'''
Třída pro implementaci architektury modelu UNet.
Vstupy:
    Počet výstupních kanálů - třídy, které rozlišuje
Výstup:
    Predikce pro vstupní obrázek
'''
class UNet(nn.Module):
    def __init__(self, out_channels=5):
        super().__init__()

        # Max pooling mezi úrovněmi
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        self.donw_conv_1 = double_conv(1, 64)
        self.donw_conv_2 = double_conv(64, 128)
        self.donw_conv_3 = double_conv(128, 256)
        self.donw_conv_4 = double_conv(256, 512)
        self.donw_conv_5 = double_conv(512, 1024)

        # Up part
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)

        # Výstupní vrstva
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)



    def forward(self, input):
        # Down part:
        # Level 5
        x1 = self.donw_conv_1(input) # Uloží se do skip connection
        x2 = self.max_pool_2x2(x1)

        # Level 4
        x3 = self.donw_conv_2(x2) # Uloží se do skip connection
        x4 = self.max_pool_2x2(x3)

        # Level 3
        x5 = self.donw_conv_3(x4) # Uloží se do skip connection
        x6 = self.max_pool_2x2(x5)

        # Level 2
        x7 = self.donw_conv_4(x6) # Uloží se do skip connection
        x8 = self.max_pool_2x2(x7)

        # Level 1
        x9 = self.donw_conv_5(x8)

        # Up part:
        # Level 1
        x = self.up_trans_1(x9)
        y = crop_tensor(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1)) # Spojí se s Level 2

        # Level 2
        x = self.up_trans_2(x)
        y = crop_tensor(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1)) # Spojí se s Level 3

        # Level 3
        x = self.up_trans_3(x)
        y = crop_tensor(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1)) # Spojí se s Level 4

        # Level 5
        x = self.up_trans_4(x)
        y = crop_tensor(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1)) # Spojí se s Level 5

        # Výstupní vrstva
        x = self.out(x)
        return x

