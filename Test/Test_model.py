import os
import glob
import torch
import numpy as np
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from Training.Utils import load_checkpoint
from UNet_architecture.UNet import UNet
from UNet_architecture.NestedUnet import NestedUNet
from Training.Dataset import ClassDataset


RESULTS_PATH = "Results.txt"


# Převod obrázku na long tensor
def to_long_tensor(img):
    return torch.from_numpy(np.array(img, dtype=np.uint8)).long()

'''
Třída pro otestování modelu na validční datové sadě.
Vstupy:
    Cesta k validační sadě
    Cesta k souboru s modelem
    Počet klasifikovatelných tříd
    Velikost vzorku batch
    Pin memory flag
    Příznak pro nastavení architektury
Výstup:
    Soubor s hodnotami metrik
'''
class ClassTestModel():
    def __init__(self, validation_set_path, model_path, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT,
                 classes_num=5, batch_size=1, pin_memory=True, nested_unet=False):
        self.validation_set_path = validation_set_path
        self.classes_num = classes_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if nested_unet == True:
            self.model = NestedUNet(out_channels=classes_num)
        else:
            self.model = UNet(out_channels=classes_num)

        load_checkpoint(model_path, self.model)
        self.model.to(self.device)


        image_folder = os.path.join(validation_set_path,"images")
        gt_folder = os.path.join(validation_set_path,"masks")

        # Transformace vstupů obrázků a anotací
        img_transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])

        gt_transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(to_long_tensor)
        ])

        val_ds = ClassDataset(
            image_dir=image_folder,
            mask_dir=gt_folder,
            img_transform=img_transform,
            gt_transform=gt_transform,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False,
        )


    def get_stats(self):
        # Zápis do souboru
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            f.write("=== Výsledky modelu ===\n")

        stats = {
            "accuracy": {i: 0.0 for i in range(self.classes_num)},
            "precision": {i: 0.0 for i in range(self.classes_num)},
            "sensitivity": {i: 0.0 for i in range(self.classes_num)},
            "specificity": {i: 0.0 for i in range(self.classes_num)},
            "IOU": {i: 0.0 for i in range(self.classes_num)},
            "dice": {i: 0.0 for i in range(self.classes_num)},
        }

        samples = 0
        
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Evaluating", total=len(self.val_loader)):
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)

                mask = y.to(self.device)

                # Pravděpodobnost pro jednotlivé třídy
                preds = torch.softmax(preds, dim=1)  # [B, C, H, W]
                preds_classes = torch.argmax(preds, dim=1)  # [B, H, W]

                for cls in range(self.classes_num):
                    # Počty pixelů
                    TP, TN, FP, FN = self.get_pixels_stats(preds_classes, mask, cls)
                    # Metriky
                    accuracy = self.get_accuracy(TP, TN, FP, FN)
                    precision = self.get_precision(TP, FP)
                    sensitivity = self.get_sensitivity(TP, FN)
                    specificity = self.get_specificity(TN, FP)
                    IOU = self.get_IOU(preds_classes, mask, cls)
                    dice = self.get_dice(preds_classes, mask, cls)

                    stats["accuracy"][cls] += accuracy
                    stats["precision"][cls] += precision
                    stats["sensitivity"][cls] += sensitivity
                    stats["specificity"][cls] += specificity
                    stats["IOU"][cls] += IOU
                    stats["dice"][cls] += dice

                samples += 1

        total_accuracy = 0
        total_precision = 0
        total_sensitivity = 0
        total_specificity = 0
        total_IOU = 0
        total_dice = 0
        
        # Zápis do souboru
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            for cls in range(self.classes_num):
                print("--------------------------------------------------------")
                f.write("--------------------------------------------------------\n")
                print(f"TŘÍDA {cls}:")
                f.write(f"TŘÍDA {cls}:\n")

                class_accuracy = stats["accuracy"][cls] / samples
                total_accuracy += class_accuracy
                print(f"Accuracy: {class_accuracy}")
                f.write(f"Accuracy: {class_accuracy}\n")

                class_precision = stats["precision"][cls] / samples
                total_precision += class_precision
                print(f"Precision: {class_precision}")
                f.write(f"Precision: {class_precision}\n")

                class_sensitivity = stats["sensitivity"][cls] / samples
                total_sensitivity += class_sensitivity
                print(f"Sensitivity: {class_sensitivity}")
                f.write(f"Sensitivity: {class_sensitivity}\n")

                class_specificity = stats["specificity"][cls] / samples
                total_specificity += class_specificity
                print(f"Specificity: {class_specificity}")
                f.write(f"Specificity: {class_specificity}\n")

                class_IOU = stats["IOU"][cls] / samples
                total_IOU += class_IOU
                print(f"IOU: {class_IOU}")
                f.write(f"IOU: {class_IOU}\n")

                class_dice = stats["dice"][cls] / samples
                total_dice += class_dice
                print(f"Dice: {class_dice}")
                f.write(f"Dice: {class_dice}\n")

            print("--------------------------------------------------------")
            f.write("--------------------------------------------------------\n")
            print("--------------------------------------------------------")
            f.write("--------------------------------------------------------\n")
            total_accuracy = total_accuracy / (self.classes_num)
            print(f"Total accuracy: {total_accuracy}")
            f.write(f"Total accuracy: {total_accuracy}\n")

            total_precision = total_precision / (self.classes_num)
            print(f"Total precision: {total_precision}")
            f.write(f"Total precision: {total_precision}\n")

            total_sensitivity = total_sensitivity / (self.classes_num)
            print(f"Total sensitivity: {total_sensitivity}")
            f.write(f"Total sensitivity: {total_sensitivity}\n")

            total_specificity = total_specificity / (self.classes_num)
            print(f"Total specificity: {total_specificity}")
            f.write(f"Total specificity: {total_specificity}\n")

            total_IOU = total_IOU / (self.classes_num)
            print(f"Total IOU: {total_IOU}")
            f.write(f"Total IOU: {total_IOU}\n")

            total_dice = total_dice / (self.classes_num)
            print(f"Total dice: {total_dice}")
            f.write(f"Total dice: {total_dice}\n")
            print("--------------------------------------------------------")
            f.write("--------------------------------------------------------")





    def get_pixels_stats(self, prediction, mask, class_value):
        prediction = prediction.cpu().numpy()
        mask = mask.cpu().numpy()

        TP = np.sum((prediction == class_value) & (mask == class_value))
        TN = np.sum((prediction != class_value) & (mask != class_value))
        FP = np.sum((prediction == class_value) & (mask != class_value))
        FN = np.sum((prediction != class_value) & (mask == class_value))

        return TP, TN, FP, FN

    
    def get_accuracy(self, TP, TN, FP, FN):
        correct = TP + TN
        total = TP + TN + FP + FN
        accuracy = correct / total if total > 0 else 0
        return accuracy

    def get_precision(self, TP, FP):
        total = TP + FP
        precision = TP / total if total > 0 else 0
        return precision


    def get_sensitivity(self, TP, FN):
        total = TP + FN
        sensitivity = TP / total if total > 0 else 0
        return sensitivity


    def get_specificity(self, TN, FP):
        total = TN + FP
        specificity = TN / total if total > 0 else 0
        return specificity

        
    def get_IOU(self, prediction, mask, class_value):
        smooth = 1e-6
        pred_mask = (prediction == class_value).float()
        true_mask = (mask == class_value).float()
        intersection = (pred_mask * true_mask).sum()
        union = pred_mask.sum() + true_mask.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    def get_dice(self, prediction, mask, class_value):
        smooth=1e-6
        pred_mask = (prediction == class_value).float()
        true_mask = (mask == class_value).float()
        intersection = (pred_mask * true_mask).sum()
        union = pred_mask.sum() + true_mask.sum()
        dice = (2 * intersection + smooth) / (union + smooth)

        return dice

    
    def find_groundtruths(image_path, gt_folder):
        # Získání čísla obrázku
        base_name = os.path.basename(image_path).split('.')[0]

        # Nalezení příslušné masky
        gt_path = glob.glob(os.path.join(gt_folder, f"{base_name}_*"))
        return gt_path[0] if gt_path else None
