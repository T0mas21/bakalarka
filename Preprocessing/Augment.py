'''
  Název souboru: Augment.py
  Autor: Tomáš Janečka
  Datum: 2025-05-04
  Popis: Umělé rozšíření datové sady
'''

import os
import numpy as np
import random
import glob
import re
from PIL import Image
from tqdm import tqdm

'''
Třída pro rozšíření datové sady. Používá metody rotace a škálování.
Vstupy:
    Cesta k adresáři s obrázky
    Cesta k adresáři s anotacemi
    Slovník {hodnota třídy : počet augmentací}
Výstup:
    Uloží nové upravené obrázky a anotace do zadaných adresářů
'''
class ClassAugment():
    def __init__(self, img_dir, gt_dir, classes_to_augment={}):
        # Cesty k aresářům
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.classes_to_augment = classes_to_augment

        max_id = -1
        # Regex pro hledání souborů např. "45.png", "1234.png"
        pattern = re.compile(r"^(\d+)\.png$") 

        # Najde poslední obrázek - obrázek s nejvyšším číslem, aby se mohli přidávat další
        for filename in os.listdir(img_dir):
            match = pattern.match(filename)
            if match:
                file_id = int(match.group(1))
                max_id = max(max_id, file_id)
        
        # Číslo pro nové obrázky
        self.id = max_id + 1

    def augment(self):
        # Projdou se všechny anotace pro zjištění hodnot tříd k augmentaci
        gt_filenames = list(os.listdir(self.gt_dir))
        for filename in tqdm(gt_filenames, desc="Augmentation"):
            gt_path = os.path.join(self.gt_dir, filename)
            gt_img = Image.open(gt_path).convert("L")
            gt_array = np.array(gt_img)
            unique_values = np.unique(gt_array)

            currently_augmented_values = []
            augment_flag = False

            # Kontrola jestli se v anotaci nachází třída k augmentaci (nesmí mít vyčerpaný počet augmentací)
            for value in unique_values:
                if value in self.classes_to_augment and self.classes_to_augment[value] > 0:
                    self.classes_to_augment[value] -= 1
                    augment_flag = True
                    currently_augmented_values.append(value)
            
            if augment_flag == False:
                continue
            else:
                # Najde se příslušný obrázek ke zpracovávané masce
                image_id = filename.split("_")[0]
                pattern = os.path.join(self.img_dir, image_id + ".*")
                matches = glob.glob(pattern)
                if not matches:
                    continue

                image_path = matches[0]
                try:
                    image = Image.open(image_path).convert("L")
                except Exception as e:
                    continue

                # Rotace o 120°
                rotated_img = self.rotate(image, angle=120)
                rotated_img.save(f'{self.img_dir}/{self.id}.png')
                        
                rotated_gt = self.rotate(gt_img, angle=120)
                rotated_gt.save(f'{self.gt_dir}/{self.id}_mask.png')

                self.id += 1

                # Kontrola jestli je možné provést další augmentace ( třída nesmí mít vyčerpaný počet augmentací)
                augment_flag = False
                for val in currently_augmented_values:
                    if self.classes_to_augment[val] > 0:
                        augment_flag = True
                
                if augment_flag == False:
                    continue
                

                # Škálování
                scale_factor = random.uniform(1.2, 1.7)

                jittered_img = self.random_scale_jittering(image, scale_factor)
                jittered_img.save(f'{self.img_dir}/{self.id}.png')

                jittered_gt = self.random_scale_jittering(gt_img, scale_factor)
                jittered_gt.save(f'{self.gt_dir}/{self.id}_mask.png')

                self.id += 1

                # Kontrola jestli je možné provést další augmentace ( třída nesmí mít vyčerpaný počet augmentací)
                augment_flag = False
                for val in currently_augmented_values:
                    if self.classes_to_augment[val] > 0:
                        augment_flag = True
                
                if augment_flag == False:
                    continue

                # Škálování + rotace o 240°
                rotated_img = self.rotate(jittered_img, angle=240)
                rotated_img.save(f'{self.img_dir}/{self.id}.png')

                rotated_gt = self.rotate(jittered_gt, angle=240)
                rotated_gt.save(f'{self.gt_dir}/{self.id}_mask.png')

                self.id += 1


    def random_scale_jittering(self, image, scale_factor):
        w, h = image.size
        
        # Rescale
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        if scale_factor > 1.0:
            # Pokud je obraz větší, převede se na původní velikost
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            cropped = resized.crop((left, top, left + w, top + h))
            return cropped
        else:
            pad_w = (w - new_w) // 2
            pad_h = (h - new_h) // 2
            new_image = Image.new(image.mode, (w, h))
            new_image.paste(resized, (pad_w, pad_h))
            return new_image
            
    
    def rotate(self, img, angle):
        # Rotace obrázku
        rotated_img = img.rotate(angle)
        return rotated_img