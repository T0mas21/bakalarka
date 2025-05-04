'''
  Název souboru: Treshold.py
  Autor: Tomáš Janečka
  Datum: 2025-05-04
  Popis: Vyprahování obrázku
'''

'''
Třída pro vyprahování obrázku. Vše nad horní hranici nastaví na 255 a vše pod spodní hranici nastaví na 0.
Vstupy:
    Obrázek
    Horní hranice
    Spodní hranice
Výstup:
    Vyprahovaný obrázek
'''
class ClassTreshold():
    @staticmethod
    def treshold(image, high, low):
        processed = image.copy()
        processed[image > high] = 255
        processed[image < low] = 0

        return processed