'''
  Název souboru: Split.py
  Autor: Tomáš Janečka
  Datum: 2025-05-04
  Popis: Ořezání pozadí a rozdělení obrázků na více částí
'''

import cv2 as cv

'''
Třída pro oříznutí obrázku. Používá metodu největšího obrysu v obrázku.
Vstupy:
    Metoda crop:
        Cesta k obrázku
        Cesta k anotaci obrázku
        Flag pro převod obrázku na RGB formát
    Metoda split_image:
        Obrázek
        Anotace
Výstup:
    Metoda crop:
        Oříznutý obrázek
        Oříznutá anotace
    Metoda split_image:
        Rozdělený obrázek
        Rozdělená anotace
'''
class ClassSplit():
    @staticmethod
    def crop(image_path, gt_path=None, rgb_flag=False):
        # Načíst obrázek
        img = cv.imread(image_path)

        # Převod do grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Gaussovo rozmazání pro vyhlazení okrajů a odstranění šumu
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # Prahování pro zvýraznění sítnice
        _, binary = cv.threshold(blurred, 10, 255, cv.THRESH_BINARY)

        # Nalezení obrysů v binárník obrázku
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Nalezení největšího obrysu
        largest_contour = max(contours, key=cv.contourArea)

        # Ohraničení kolem obrysu
        x, y, w, h = cv.boundingRect(largest_contour)

        # Oříznutí původního obrázku podle obrysu
        cropped_img = img[y:y+h, x:x+w]
        
        # V případě tohoto flagu se vrátí barevný obrázek
        if rgb_flag == True:
            cropped_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB)   

        if gt_path is not None:
            # Oříznutí anotace musí být stejné jako u obrázku 
            gt_img = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
            cropped_gt = gt_img[y:y+h, x:x+w]

            return cropped_img, cropped_gt
        
        return cropped_img, None
    
    @staticmethod
    def split_image(image, mask):
        # Rozměry částí obrázků
        img_height, img_width = image.shape[:2]
        patch_size_h = img_height // 2
        patch_size_w = img_width // 2

        # Části obrázku a masky
        patches = []
        mask_patches = []

        # Velikost kroku
        step_h = patch_size_h
        step_w = patch_size_w

        # Rozdělení obrázku
        for y in range(0, img_height - patch_size_h + 1, step_h):
            for x in range(0, img_width - patch_size_w + 1, step_w):

                img_patch = image[y:y + patch_size_h, x:x + patch_size_w]
                patches.append(img_patch)

                mask_patch = mask[y:y + patch_size_h, x:x + patch_size_w]
                mask_patches.append(mask_patch)

        return patches, mask_patches