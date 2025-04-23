import cv2 as cv
import os
import glob
from tqdm import tqdm 

from Preprocessing.Split import ClassSplit
from Preprocessing.Filter import ClassFilter
from Preprocessing.Augment import ClassAugment
from Preprocessing.Treshold import ClassTreshold


'''
Třída pro předzpracování datasetu. Používá třídy ClassSplit, ClassFilter a ClassAugment.
Vstupy:
    Cesta k adrsáři původního datasetu
    Slovník {hodnota třídy : počet augmentací}
Výstup:
    Nový předzpracovaný dataset
'''
class ClassPreprocesing:
    @staticmethod
    def preprocess_dataset(dataset_root, classes_to_augment={}, treshold_high=255, treshold_low=0, split_flag=False):
        # Každý obrázek má unikátní číslo
        image_id = 0

        # Vytvoření nového předzpracovaného datasetu
        base_path = "preprocesed_dataset"
        os.makedirs(os.path.join(base_path, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "train", "masks"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "validate", "images"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "validate", "masks"), exist_ok=True)
        
        for subset in ["train", "validate"]:

            path_new = os.path.join(base_path, subset)

            # Obrázky a příslušné masky
            image_folder = os.path.join(dataset_root, subset, "images")
            gt_folder = os.path.join(dataset_root, subset, "masks")

            # Přeskočit neexistující složky
            if not os.path.exists(image_folder) or not os.path.exists(gt_folder):
                print(f"Skipping {subset} as folders do not exist.\n")
                continue


            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            image_paths = [f for f in glob.glob(os.path.join(image_folder, "*.*")) if f.lower().endswith(valid_extensions)]

            # Zpracování každého obrázku
            for image_path in tqdm(image_paths, desc=f"Processing {subset} images"):
                gt_path = ClassPreprocesing.find_groundtruths(image_path, gt_folder)

                # Pokud se nenajde příslušná maska obrázek se přeskočí
                if not gt_path:
                    print(f"Warning: No ground truths found for {image_path}")
                    continue
                
                new_img_path = os.path.join(path_new, "images")
                new_gt_path = os.path.join(path_new, "masks")

                # Předzpracování
                # 1. ořítznutí obrázku a jeho anotace
                cropped_img, cropped_gt = ClassSplit.crop(image_path, gt_path)

                # CLAHE + Gaborův filtr
                filtered_img = ClassFilter.filter(cropped_img)

                # Prahování filtrovaného obrázku
                if treshold_low > 0 or treshold_high < 255:
                    filtered_img = ClassTreshold.treshold(image=filtered_img, high=treshold_high, low=treshold_low)

                # Rozdělení obrázků na více částí (na 4)
                if split_flag == True:
                    patches, mask_patches = ClassSplit.split_image(filtered_img, cropped_gt)
                    for i in range(0, len(patches)):
                        cv.imwrite(f"{new_img_path}/{image_id}.png", patches[i])
                        cv.imwrite(f"{new_gt_path}/{image_id}_mask.png", mask_patches[i])
                        image_id += 1

                else:
                    # Uložit předzpracovaný obrázek
                    cv.imwrite(f"{new_img_path}/{image_id}.png", filtered_img)
                    cv.imwrite(f"{new_gt_path}/{image_id}_mask.png", cropped_gt)

                    image_id += 1

        # Pokud jsou zadané třídy k augmentaci, augmentuje se trénovací část
        if len(classes_to_augment) > 0:
            img_dir = os.path.join(base_path, "train", "images")
            gt_dir = os.path.join(base_path, "train", "masks")
            augmentor = ClassAugment(img_dir=img_dir, gt_dir=gt_dir, classes_to_augment=classes_to_augment)
            augmentor.augment()


    @staticmethod
    def find_groundtruths(image_path, gt_folder):
        # Získání čísla obrázku
        base_name = os.path.basename(image_path).split('.')[0]

        # Nalezení příslušné masky
        gt_path = glob.glob(os.path.join(gt_folder, f"{base_name}_*"))
        return gt_path[0] if gt_path else None