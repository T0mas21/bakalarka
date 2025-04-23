import torch
import torch.nn as nn
import torch.nn.functional as F


# Zdroj: https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9

'''
Funkce pro více-třídní focal loss s alfa váhami pro jednotlivé třídy
Vstupy:
    Výstup modelu
    Anotace
    Alfa váhy
Výstup:
    Hodnota ztrátové funkce
'''
def focal_loss_multiclass_tensor(inputs, targets, alpha=[0.03, 2.2, 6.0, 10.0, 2.15], gamma=2.5):
    # Převod na pravděpodobnostní výstup
    log_prob = F.log_softmax(inputs, dim=1)
    prob = torch.exp(log_prob)

    # One-hot enkódování
    num_classes = inputs.shape[1]
    targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)

    assert targets.max() < num_classes, (
        f"Hodnota v targets ({targets.max().item()}) >= num_classes ({num_classes})"
    )
    assert targets.min() >= 0, (
        f"Hodnota v targets ({targets.min().item()}) < 0 – očekávají se nezáporné třídy"
    )

    # Přeuspořádání: [B, H, W, C] na [B, C, H, W]
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    # Pravděpodobnost pro správnou tříd
    pt = (prob * targets_one_hot).sum(dim=1)  # [B, H, W]

    # Váha dané třídy na pozici
    alpha = torch.tensor(alpha, dtype=torch.float32, device=inputs.device)
    alpha_t = alpha[targets]  # [B, H, W]


    # Výpočet focal loss
    log_prob_true = (log_prob * targets_one_hot).sum(dim=1)  # [B, H, W]
    focal_loss = -alpha_t * (1 - pt) ** gamma * log_prob_true  # [B, H, W]

    return focal_loss.mean()



'''
Třída pro více-třídní dice loss + focal loss s váhami pro jednotlivé třídy
Vstupy:
    Výstup modelu
    Anotace
    Váhy
Výstup:
    Celková hodnota ztrátové funkce
'''
class HybridLoss_multiclass(nn.Module):
    def __init__(self, weights=[0.05, 2.5, 2.5, 4.0, 2.0]):
        super(HybridLoss_multiclass, self).__init__()
        self.focal_loss = focal_loss_multiclass_tensor
        self.weights = weights

    def forward(self, inputs, targets):
        # Focal Loss
        focal_loss = self.focal_loss(inputs, targets, alpha=self.weights)

        # Dice Loss
        num_classes = inputs.shape[1]
        prob = F.softmax(inputs, dim=1)
        smooth = 1e-6

        # Převedení: [B, H, W] na [B, C, H, W]
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Výpočet průniků pro každou třídu
        intersection = (prob * targets_one_hot).sum(dim=(0, 2, 3))
        union = prob.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        dice_score = (2 * intersection + smooth) / (union + smooth)

        # Ztráta pro každou třídu
        dice_loss_per_class = 1 - dice_score

        alpha = torch.tensor(self.weights, device=inputs.device)
        dice_loss = (dice_loss_per_class * alpha).sum() / alpha.sum()

        return 0.5 * focal_loss + 0.5 * dice_loss
    

