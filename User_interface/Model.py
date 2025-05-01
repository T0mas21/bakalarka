import torch
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import supervision as sv

from Preprocessing.Split import ClassSplit
from Preprocessing.Filter import ClassFilter
from UNet_architecture.UNet import UNet
from UNet_architecture.NestedUnet import NestedUNet
from Preprocessing.Treshold import ClassTreshold
from Training.Utils import load_checkpoint


'''
Třída pro backend pro uživatelské rozhraní.
Vstupy:
    Nahraný obrázek
    Nahrný model
Výstup:
    Predikce pro vstupní obrázek
'''
class ClassModel():
    def __init__(self, IMAGE_HEIGHT, IMAGE_WIDTH, treshold_high=255, treshold_low=0, nested_unet=False):
        self.original_image = None
        self.preprocesed_image = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction = None
        self.prediction_image = None
        self.treshold_high = treshold_high
        self.treshold_low = treshold_low
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.nested_unet = nested_unet

    def set_image(self, image_path):
        # Načtení obrázku
        self.original_image = image_path
        self.original_image = Image.open(image_path)

        # Předzpraconání obrázku
        croped_image, _ = ClassSplit.crop(image_path)
        filtered_image = ClassFilter.filter(croped_image)
        if self.treshold_low > 0 or self.treshold_high < 255:
            filtered_image = ClassTreshold.treshold(image=filtered_image, high=self.treshold_high, low=self.treshold_low) 
        self.preprocesed_image = Image.fromarray(filtered_image)
        self.preprocesed_image = self.preprocesed_image.resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), Image.LANCZOS)

        # Původní oříznutý obrázek pro vizualizaci
        croped_image_rgb, _ = ClassSplit.crop(image_path, rgb_flag=True)
        pil_croped = Image.fromarray(croped_image_rgb)
        self.original_image = pil_croped.resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), Image.LANCZOS)

    def set_model(self, model_path, classes_num=5):
        # Načtení modelu
        if self.nested_unet == True:
            self.model = NestedUNet(out_channels=classes_num)
        else:
            self.model = UNet(out_channels=classes_num)
        load_checkpoint(model_path, self.model)
        self.model.to(self.device)

    
    def get_results(self):
        if self.model is not None and self.preprocesed_image is not None:
            # Transformace na tensor
            img_transform = transforms.Compose([
                transforms.Resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH)),
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
            original_np = np.array(self.original_image.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))).astype(np.uint8)

            # Vytvoření RGB masky ve stejné velikosti jako predikovaná maska
            h, w = pred_mask.shape
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for label, color in color_map.items():
                rgb_mask[pred_mask == label] = color

            # Resize masky na stejnou velikost jako originál (kvůli kompatibilitě)
            rgb_mask_resized = cv2.resize(rgb_mask, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
            pred_mask_resized = cv2.resize(pred_mask, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

            self.prediction = pred_mask_resized

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

            # Unikátní hodnoty z predikce, pro informaci zda je pozitivní nebo ne
            unique_labels = np.unique(pred_mask)
            unique_labels = unique_labels[unique_labels != 0]

            self.prediction_image = blended

            return blended, unique_labels

        return None, []
    
    def get_boundingbox(self, margin: int = 5):
        def remove_nested_boxes(xyxy_list, class_ids, confidences):
            keep = []
            for i, (x1a, y1a, x2a, y2a) in enumerate(xyxy_list):
                is_inside = False
                for j, (x1b, y1b, x2b, y2b) in enumerate(xyxy_list):
                    if i == j:
                        continue
                    # kontrolujeme pouze boxy stejné třídy
                    if class_ids[i] != class_ids[j]:
                        continue
                    if x1a >= x1b and y1a >= y1b and x2a <= x2b and y2a <= y2b:
                        is_inside = True
                        break
                if not is_inside:
                    keep.append(i)
            return (
                [xyxy_list[i] for i in keep],
                [class_ids[i] for i in keep],
                [confidences[i] for i in keep]
            )

        if self.prediction is not None:
            pred_mask = self.prediction
            h, w = pred_mask.shape
            xyxy_list = []
            class_ids = []
            confidences = []

            for label in np.unique(pred_mask):
                if label == 0:
                    continue
                
                mask = (pred_mask == label).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    x, y, bw, bh = cv2.boundingRect(contour)
                    x_min = max(x - margin, 0)
                    y_min = max(y - margin, 0)
                    x_max = min(x + bw + margin, w)
                    y_max = min(y + bh + margin, h)

                    xyxy_list.append([x_min, y_min, x_max, y_max])
                    class_ids.append(label)
                    confidences.append(1.0)
            
            if len(class_ids) == 0:
                return None

            # Odstranit vnořené boxy
            xyxy_list, class_ids, confidences = remove_nested_boxes(xyxy_list, class_ids, confidences)

            if not xyxy_list:
                print("No bounding boxes found.")
                return self.original_image

            detections = sv.Detections(
                xyxy=np.array(xyxy_list),
                class_id=np.array(class_ids),
                confidence=np.array(confidences)
            )



            bgr_colors = [sv.Color(0, 0, 0), sv.Color(0, 0, 255), sv.Color(0, 255, 0),
                          sv.Color(255, 0, 0), sv.Color(0, 255, 255), sv.Color(255, 0, 255),
                          sv.Color(255, 255, 0), sv.Color(255, 255, 255)]
            

            custom_palette = sv.ColorPalette(colors=bgr_colors)


            annotator = sv.BoxAnnotator(
                color=custom_palette,      # automatická sada barev
                color_lookup=sv.ColorLookup.CLASS  # podle class_id
            )

            annotated_image = annotator.annotate(
                scene=np.array(self.original_image.resize((w, h))).copy(),
                detections=detections,
            )

            return annotated_image

    def get_prediction(self):
        return self.prediction_image