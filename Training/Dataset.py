import os
import torch
from PIL import Image
from torch.utils.data import Dataset

'''
Třída pro načítání obrázků a anotací z datasetu.
Vstupy:
    Cesta k adresáři obrázků
    Cesta k adresáři anotací
    Transfomrace obrázků
    Transfomrace anotací
    Index souboru
Výstup:
    Transformovaný obrázek
    Transformovaná anotace
'''
class ClassDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None, gt_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        # Seznam obrázků
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    def __len__(self):
        # Velikost datasetu
        return len(self.images) 
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            # Načíst obrázek
            image = Image.open(img_path).convert("L")
            image = image.resize((572, 572), Image.LANCZOS)

            # Načíst masku
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((388, 388), Image.NEAREST)

            # Aplikace transformací
            if self.img_transform is not None:
                image = self.img_transform(image)
            if self.gt_transform is not None:
                mask = self.gt_transform(mask).squeeze(0)

            return image, mask
        
        except Exception as e:
            # Pokud se soubor nepodařilo otevřít nebo je poškozen vrátí se nulové tensory 
            dummy_image = torch.zeros((1, 572, 572), dtype=torch.float32)
            dummy_mask = torch.zeros((388, 388), dtype=torch.long)
            return dummy_image, dummy_mask