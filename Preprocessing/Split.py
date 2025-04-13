import cv2 as cv

'''
Třída pro oříznutí obrázku. Používá metodu největšího obrysu v obrázku.
Vstupy:
    Cesta k obrázku
    Cesta k anotaci obrázku
    Flag pro převod obrázku na RGB formát
Výstup:
    Oříznutý obrázek
    Oříznutá anotace
'''
class ClassSplit():

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
    