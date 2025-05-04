'''
  Název souboru: Train.py
  Autor: Tomáš Janečka
  Datum: 2025-05-04
  Popis: Implementace procesu trénování
'''

import torch
from tqdm import tqdm
import torch.optim as optim
import os
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import numpy as np

from UNet_architecture.UNet import UNet
from UNet_architecture.NestedUnet import NestedUNet
from UNet_architecture.UNet import init_weights_he
from Training.Loss_function import HybridLoss_multiclass

from Training.Utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    evaluate_multiclass_dice,
    save_dice_metrics,
)





# Převod obrázku na long tensor
def to_long_tensor(img):
    return torch.from_numpy(np.array(img, dtype=np.uint8)).long()

'''
Třída pro trénování modelu. Používá architekturu UNet a ztrátovou funkci z třídy HybridLoss_multiclass
Vstupy:
    Cesta k trénovacímu adresáři s obrázky
    Cesta k trénovacímu adresáři s anotacemi
    Cesta k validačnímu adresáři s obrázky
    Cesta k validačnímu adresáři s anotacemi
    Váhy pro jednotlivé třídy
    Cesta k načtení předtrénovaného modelu
    Počet tříd
    Learning rate
    Batch size 
    Počet epoch
    Pin memory flag
    Počet epoch, po kterých se při nezlepšení dice skóre trénování zastaví
    Příznak pro nastavení architektury
Výstup:
    Natrénovaný model
'''
class ClassTrain():
    def __init__(self, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT,
                 weights=[], model_path=None, classes_num=5, learning_rate=1e-3, batch_size=1, num_epoch=100, pin_memory=True, early_stop=10, nested_unet=False):
        
        # Nastavení hyperparametrů
        self.train_img_dir = train_img_dir
        self.train_mask_dir = train_mask_dir
        self.val_img_dir = val_img_dir
        self.val_mask_dir = val_mask_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.pin_memory = pin_memory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Transformace vstupů obrázků a anotací
        self.img_transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(to_long_tensor)
        ])

        # Statistiky
        self.total_loss = []
        self.all_epoch_dice_mean = []
        self.total_val_loss = []
        self.output_dir = "UNet_results"
        os.makedirs(f"{self.output_dir}/graphs", exist_ok=True)
        self.epoch_number = 1
        self.best_flag = 0
        self.best_dice_score = 0.0
        self.epochs_since_improvement = 0

        # Model
        if nested_unet == True:
            self.model = NestedUNet(out_channels=classes_num)
        else:
            self.model = UNet(out_channels=classes_num)
        if model_path is None:
            self.model.apply(init_weights_he)
        self.model = self.model.to(self.device).float()

        # Optimalizační algoritmus
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)

        if model_path is not None:
            self.best_dice_score = load_checkpoint(model_path, self.model, self.optimizer)

        self.classes_num = classes_num
        # Váhy pro třídy modelu
        self.weights = []
        # Pokud nejsou zadané váhy pro třídy, pak se inicializují všechny na 1.0
        if len(weights) == 0:
            for i in range(0, classes_num):
                self.weights.append(1.0)
        else:
            self.weights = weights

        # Ztrátová funkce
        self.loss_fn = HybridLoss_multiclass(weights=self.weights)

        # Data loader
        self.train_loader, self.val_loader = get_loaders(
            self.train_img_dir,
            self.train_mask_dir,
            self.val_img_dir,
            self.val_mask_dir,
            self.batch_size,
            self.img_transform,
            self.gt_transform,
            self.pin_memory,
        )

        # Scaler
        self.scaler = torch.amp.GradScaler(device=self.device) # Faster training on GPU

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6, 
            verbose=True
        )

        # Předčasné ukončení trénování
        self.early_stop = early_stop



    def train(self):
        # Progress bar pro vizualizaci
        loop = tqdm(self.train_loader)
        # Počet vzorků v jedné epoše
        num_batches = len(self.train_loader)
        batch_loss = 0
        
        for batch_idx, (data, targets) in enumerate(loop):
            # Přesun vstupních dat a anotací na stejné zařízení
            data = data.to(device=self.device)
            targets = targets.long().to(device=self.device)

            # Forward
            with torch.amp.autocast(device_type="cuda"):
                predictions = self.model(data)

            predictions = predictions.float()
            loss = self.loss_fn(predictions, targets)

            # Backward
            # Vynulování gradientů
            self.optimizer.zero_grad()
            # Backpropagace
            self.scaler.scale(loss).backward()
            # Aktualizace vah
            self.scaler.step(self.optimizer)
            # Aktualizace škálovacího faktoru
            self.scaler.update()

            batch_loss += loss.item()

            # Zobraz aktuální hodnoty v progress bar
            loop.set_postfix(loss=loss.item())
        
        # Průměrná ztráta
        epoch_loss = batch_loss / num_batches
        batch_loss = 0
        self.total_loss.append(epoch_loss)
        print("Epoch loss = ", epoch_loss)


    def run(self):
        for epoch in range(self.num_epoch):
            self.train()

            # Vyhodnocení modelu na validační sadě
            dice_per_class, val_epoch_loss = evaluate_multiclass_dice(self.val_loader, self.model, self.loss_fn, device=self.device, num_classes=self.classes_num)
            self.all_epoch_dice_mean.append(dice_per_class.mean().item())
            self.total_val_loss.append(val_epoch_loss)
            
            # Uložení statistik validace
            save_dice_metrics(self.output_dir, self.all_epoch_dice_mean, self.total_loss, self.total_val_loss)

            # Úprava scheduler, který pak upravuje learning rate, podle validační ztráty 
            self.scheduler.step(val_epoch_loss)
            
            # Vyhodnocení nejlepšího výkonu
            if (len(self.all_epoch_dice_mean) == 0) or (len(self.all_epoch_dice_mean) > 0 and self.all_epoch_dice_mean[-1] > self.best_dice_score):
                self.best_dice_score = self.all_epoch_dice_mean[-1]
                self.best_flag = 1
                self.epochs_since_improvement = 0

            # Uložení 
            checkpoint = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_dice_score": self.best_dice_score,
            }

            save_checkpoint(checkpoint, filename=f"{self.output_dir}/checkpoint{self.epoch_number}_{self.best_flag}.pth.tar")
            self.epoch_number += 1
            self.best_flag = 0
            
            
            # Předčasné ukončení trénování
            self.epochs_since_improvement += 1
            if self.epochs_since_improvement >= self.early_stop:
                print(f"Dice score has not improved for {self.early_stop} consecutive epochs — early stopping.")
                break

