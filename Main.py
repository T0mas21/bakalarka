'''
  Název souboru: Main.py
  Autor: Tomáš Janečka
  Datum: 2025-05-04
  Popis: Načtení a zpracování argumentů a sputění programu
'''

import torch.multiprocessing as mp
import tkinter as tk
import argparse
import json
import os
import ast

from Preprocessing.Preproces import ClassPreprocesing
from Training.Train import ClassTrain
from User_interface.View import ClassView
from Test.Test_model import ClassTestModel



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DATASET = "dataset"

# Cesty k datasetům
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "work_dataset/train/images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "work_dataset/train/masks")
VAL_IMG_DIR = os.path.join(BASE_DIR, "work_dataset/validate/images")
VAL_MASK_DIR = os.path.join(BASE_DIR, "work_dataset/validate/masks")
VAL_DIR = os.path.join(BASE_DIR, "work_dataset/validate")


# Parametry pro běh programu
def get_args():
    parser = argparse.ArgumentParser(description="Trénink UNet modelu")

    group = parser.add_mutually_exclusive_group(required=True)

    # První možnost preprocesing datasetu
    # Flag pro předzpracování
    group.add_argument('--preproces', action='store_true', help="Pokud je přepínač zadán, dataset se předzpracuje (povinný, přepínač)")
    # Počet augmentací ke každé třídě
    parser.add_argument('--class_augment', type=str, default="{}", help='Třídy a počet augmentací, např. --class_augment \'{"1": 40, "2": 50}\' (volitelný, str, default {})')
    parser.add_argument('--treshold_low', default=0, type=int, help='Dolní práh (volitelný, int, default 0)')
    parser.add_argument('--treshold_high', default=255, type=int, help='Horní práh (volitelný, int, default 255)')
    parser.add_argument('--split', default=False, type=bool, help='Rozdělení obrázků na více částí (volitelný, bool, default False)')

    # Druhá možnost trénovnání
    # Flag pro trénování
    group.add_argument('--train_model', action='store_true', help="Pokud je přepínač zadán, spustí se trénování modelu (povinný, přepínač)")

    # Hyperparametry
    parser.add_argument('--model_path', type=str, default=None, help='Cesta k předtrénovanému modelu (volitelný, str, default None)')
    parser.add_argument('--classes_num', type=int, default=5, help='Počet tříd pro segmentaci (volitelný, int, default 5)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (volitelný, float, default 1e-4)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (volitelný, int, default 1)')
    parser.add_argument('--num_epoch', type=int, default=100, help='Počet epoch (volitelný, int, default 100)')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Použít pin_memory u DataLoaderu (volitelný, bool, default True)')
    parser.add_argument('--early_stop', type=int, default=10, help='Při nezlepšení dicescore po daném počtu epoch ukončí trénování (volitelný, int, default 10)')

    # Váhy pro třídy
    parser.add_argument('--class_weights', type=json.loads, default=[], help='Váhy tříd jako JSON list, např. --class_weights "[1, 2.5, 2, 4.1, 1]" (volitelný, json.loads, default [])')

    # Výběr architektury modelu
    parser.add_argument('--nested_unet', type=bool, default=False, help='Použít architekturu Nested UNet (volitelný, bool, default False)')

    # Třetí možnost
    group.add_argument('--run_model', action='store_true', help="Pokud je přepínač zadán, spustí se GUI (povinný, přepínač)")

    # Čtvrtá možnost
    group.add_argument('--test_model', action='store_true', help="Pokud je přepínač zadán, spustí se testování metrik na validačním datasetu (povinný, přepínač)")

    return parser.parse_args()



if __name__ == "__main__":
    try:
        args = get_args()
        
        # Konstanty pro velikost vstupů a výstupů
        IMAGE_HEIGHT = 572 
        IMAGE_WIDTH = 572 
        IMAGE_HEIGHT_OUT = 388 
        IMAGE_WIDTH_OUT = 388 
        if args.nested_unet == True:
            IMAGE_HEIGHT = 576
            IMAGE_WIDTH = 576 
            IMAGE_HEIGHT_OUT = 576 
            IMAGE_WIDTH_OUT = 576 

        
        # Trénování
        if args.train_model:
            required_dirs = [
                TRAIN_IMG_DIR,
                TRAIN_MASK_DIR,
                VAL_IMG_DIR,
                VAL_MASK_DIR
            ]
            for path in required_dirs:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"Složka neexistuje: {path}")
            
            if args.class_weights:
                if len(args.class_weights) != args.classes_num:
                    raise ValueError(f"Počet vah pro třídy ({len(args.class_weights)}) neodpovídá počtu tříd ({args.classes_num}).")

            trainer = ClassTrain(
                # Umístění datasetu
                train_img_dir=TRAIN_IMG_DIR, train_mask_dir=TRAIN_MASK_DIR, val_img_dir=VAL_IMG_DIR, val_mask_dir=VAL_MASK_DIR,
                # Velikosti vstupu s výstupu
                IMAGE_HEIGHT=IMAGE_HEIGHT, IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT_OUT=IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT=IMAGE_WIDTH_OUT,
                # Hyperparametry
                weights=args.class_weights, model_path=args.model_path, classes_num=args.classes_num, learning_rate=args.learning_rate,
                batch_size=args.batch_size, num_epoch=args.num_epoch, pin_memory=args.pin_memory, early_stop=args.early_stop, nested_unet=args.nested_unet)
            trainer.run()

        # Předzpracování datasetu
        elif args.preproces: 
            args.class_augment = ast.literal_eval(args.class_augment)
            args.class_augment = {int(k): v for k, v in args.class_augment.items()}
            ClassPreprocesing.preprocess_dataset(ORIGINAL_DATASET, args.class_augment, args.treshold_high, args.treshold_low, args.split)
        
        # Uživatelské rozhraní
        elif args.run_model:
            root = tk.Tk()
            app = ClassView(root, IMAGE_HEIGHT, IMAGE_WIDTH, args.classes_num, args.treshold_high, args.treshold_low, args.nested_unet)
            app.start()

        # Výpočet metrik
        elif args.test_model:
            if args.model_path is None:
                raise ValueError(f"Je potřeba zadat cestu k testovanému modelu.")
            
            tester = ClassTestModel(VAL_DIR, model_path=args.model_path, classes_num=args.classes_num, batch_size=args.batch_size, pin_memory=args.pin_memory,
                                    IMAGE_HEIGHT=IMAGE_HEIGHT, IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT_OUT=IMAGE_HEIGHT_OUT, IMAGE_WIDTH_OUT=IMAGE_WIDTH_OUT, 
                                    nested_unet=args.nested_unet)
            tester.get_stats()

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C)")
    finally:
        for p in mp.active_children():
            p.terminate()