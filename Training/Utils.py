import torch
from Training.Dataset import ClassDataset
from torch.utils.data import DataLoader
import os
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    # Uložení aktuálního stavu modelu a optimalizátoru
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(path, model, optimizer=None):
    # Načtení stavu modelu a optimalizátoru
    print("Loading checkpoint...")
    checkpoint = torch.load(path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Načtení nejlepšího dosaženého Dice skóre (pokud existuje)
    best_dice_score = checkpoint.get("best_dice_score", 0.0) 
    return best_dice_score



def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, img_transform, gt_transform, pin_memory=True):
    # Trénovacího dataset
    train_ds = ClassDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        img_transform=img_transform,
        gt_transform=gt_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle = True,
    )
    
    # Validační dataset
    val_ds = ClassDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        img_transform=img_transform,
        gt_transform=gt_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def evaluate_multiclass_dice(loader, model, loss_fn, device="cuda", num_classes=5):
    # Ignorovat pozadí
    ignore_index=0
    # Vyhlazení pro stabilitu výpočtu Dice
    smooth=1e-6
    
    # Evaluation mód
    model.eval()
    all_dice = []
    batch_loss = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", total=len(loader)):
            x = x.to(device)
            y = y.to(device)
            preds = model(x)

            # Validační ztrátová funkce
            predictions = preds.float()
            targets = y.long().to(device="cuda")
            loss = loss_fn(predictions, targets)
            batch_loss += loss.item()

            # Pravděpodobnost pro jednotlivé třídy
            preds = torch.softmax(preds, dim=1)  # [B, C, H, W]
            preds_classes = torch.argmax(preds, dim=1)  # [B, H, W]

            batch_dice = []
            for cls in range(num_classes):
                if cls == ignore_index:
                    continue
                pred_mask = (preds_classes == cls).float()
                true_mask = (y == cls).float()
                intersection = (pred_mask * true_mask).sum()
                union = pred_mask.sum() + true_mask.sum()
                dice = (2 * intersection + smooth) / (union + smooth)
                batch_dice.append(dice.item())
            all_dice.append(batch_dice)

    # Trénovací mód
    model.train()
    all_dice = torch.tensor(all_dice)
    mean_per_class = all_dice.mean(dim=0)

    if len(loader) > 0:
        epoch_loss = batch_loss / len(loader)
    else:
        epoch_loss = 0

    return mean_per_class, epoch_loss



def save_dice_metrics(output_dir, all_epoch_dice_mean, total_loss, total_val_loss):
    os.makedirs(f"{output_dir}/graphs", exist_ok=True)

    epochs = list(range(1, len(all_epoch_dice_mean) + 1))
   
    # Průměrné dice skóre 
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, all_epoch_dice_mean, marker='o', linestyle='-', color='g', label='Mean Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Dice Score')
    plt.title('Mean Dice Score over Epochs')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/graphs/dice_mean_graph.png")
    plt.close()
    
    # Hodnota ztrátové funkce přes epochy 
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, total_loss, marker='o', linestyle='-', color='r', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/graphs/loss_graph.png")
    plt.close()

    # Hodnota validační ztrátové funkce přes epochy 
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, total_val_loss, marker='o', linestyle='-', color='r', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation loss')
    plt.title('Validation loss over Epochs')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/graphs/val_loss_graph.png")
    plt.close()