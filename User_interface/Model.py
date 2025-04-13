import torch
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from Preprocessing.Split import ClassSplit
from Preprocessing.Filter import ClassFilter
from UNet_architecture.UNet import UNet
from Training.Utils import load_checkpoint

IMAGE_HEIGHT = 572
IMAGE_WIDTH = 572
IMAGE_HEIGHT_OUT = 388 
IMAGE_WIDTH_OUT = 388 

'''
Třída pro backend pro uživatelské rozhraní.
Vstupy:
    Nahraný obrázek
    Nahrný model
Výstup:
    Predikce pro vstupní obrázek
'''
class ClassModel():
    def __init__(self):
        self.original_image = None
        self.preprocesed_image = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_image(self, image_path):
        # Načtení obrázku
        self.original_image = image_path
        self.original_image = Image.open(image_path)

        # Předzpraconání obrázku
        croped_image, _ = ClassSplit.crop(image_path)
        filtered_image = ClassFilter.filter(croped_image)
        self.preprocesed_image = Image.fromarray(filtered_image)
        self.preprocesed_image = self.preprocesed_image.resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.LANCZOS)

        # Původní oříznutý obrázek pro vizualizaci
        croped_image_rgb, _ = ClassSplit.crop(image_path, rgb_flag=True)
        pil_croped = Image.fromarray(croped_image_rgb)
        self.original_image = pil_croped.resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.LANCZOS)

    def set_model(self, model_path, classes_num=5):
        # Načtení modelu
        self.model = UNet(out_channels=classes_num)
        load_checkpoint(model_path, self.model)
        self.model.to(self.device)

    
    def get_results(self):
        if self.model is not None and self.preprocesed_image is not None:
            # Transformace na tensor
            img_transform = transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
            ])
            image_tensor = img_transform(self.preprocesed_image).unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Barvy pro jednotlivé třídy
            color_map = {
                0: (0, 0, 0),           # pozadí
                1: (255, 0, 0),         # Hard Exudates - červená
                2: (0, 255, 0),         # Soft Exudates - zelená
                3: (0, 0, 255),         # Microaneurysms - modrá
                4: (255, 255, 0),       # Hemorrhages - žlutá
                5: (255, 0, 255),       # Další barva pro případné třídy navíc
                6: (0, 255, 255),       # Další barva pro případné třídy navíc
                7: (255, 255, 255)      # Další barva pro případné třídy navíc
            }

            # Převedení obrázku na numpy (RGB) a přizpůsobení velikost
            original_np = np.array(self.original_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))).astype(np.uint8)

            # Vytvoření RGB masky ve stejné velikosti jako predikovaná maska
            h, w = pred_mask.shape
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for label, color in color_map.items():
                rgb_mask[pred_mask == label] = color

            # Resize masky na stejnou velikost jako originál (kvůli kompatibilitě)
            rgb_mask_resized = cv2.resize(rgb_mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
            pred_mask_resized = cv2.resize(pred_mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

            # Vyřazení pozadí
            mask_no_background = rgb_mask_resized.copy()
            mask_no_background[pred_mask_resized == 0] = [0, 0, 0]

            # Sloučení pomocí alpha blending
            alpha = 0.7
            blended = original_np.copy()
            non_black_mask = np.any(mask_no_background != [0, 0, 0], axis=-1)
            blended[non_black_mask] = (
                (1 - alpha) * blended[non_black_mask] + alpha * mask_no_background[non_black_mask]
            ).astype(np.uint8)

            return blended

        return None
