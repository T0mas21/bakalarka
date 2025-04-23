import cv2 as cv
import numpy as np


'''
Třída pro filtrování obrázku. Používá gaborův a CLAHE filtr.
Vstupy:
    Obrázek
Výstup:
    Filtrovaný obrázek
'''
class ClassFilter():
    @staticmethod
    def filter(image):
        # Převod na zelený kanál
        green_channel = image[:, :, 1]

        # Aplikace CLAHE filtru 
        clahe_image = ClassFilter.CLAHE(green_channel)

        # Aplikace gaborova filtru
        gabor_filtered_image = ClassFilter.gabor_filter(clahe_image)

        return gabor_filtered_image
    
    @staticmethod
    def CLAHE(img):
        # CLAHE filtr s nastaveným limitem klipu a velikostí dlaždice
        claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        # Aplikace filtru
        claheimg = claheObj.apply(img)

        return claheimg
    
    @staticmethod
    def gabor_filter(image, ksize=9, sigma=2, theta=np.pi/4, lambd=5, gamma=1, psi=0):

        # Vytvoření jádra pro Gaborův filtr
        gabor_kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_32F)

        # Aplikace filtru
        filtered_image = cv.filter2D(image, cv.CV_8UC1, gabor_kernel)

        return filtered_image 