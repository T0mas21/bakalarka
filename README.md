# Program pro lokalizaci příznaků diabetické retinopatie
Celkový výsledek se skládá ze scriptu pro předzpracování datové sady, scriptu pro trénování modelu architektury UNet na vlastních datech a program s uživatelským rozhraním, kde je možné nahrát model této architektury a snímek sítnice a program na výstupu ukáže výskyt jednotlivých onemocnění.

## Použití
Program pracuje s verzí `Python 3.11.9` a verzí `pip 25.0.1`. Pro použití programu je nutné nainstalovat `requirements.txt`.
```bash
pip install -r requirements.txt
```


### Předzpracování a augmentace
Vstupní dataset musí mít následující adresářovou strukturu:

<pre>
dataset/ 
├── train/ 
│   ├── images/ 
│   └── masks/ 
└── validate/ 
    ├── images/ 
    └── masks/
</pre>

Každý obrázek a jeho anotace musí mít odpovídající názvy souborů:

- **Obrázky**: `<id>.(jpg|jpeg|png|bmp|tif|tiff)`  
  Např.: `12.png`

- **Anotace**: `<id>_mask.(jpg|jpeg|png|bmp|tif|tiff)`  
  Např.: `12_mask.png`

Spuštění programu s argumenty: 
- **--preproces**: Flag pro předzpracování. Program předzpracuje dataset a uloží do adresáře `preprocesed_dataset`.
- **--class_augment**: Musí k mít sobě mít parametr slovník {hodnota třídy : maximální počet augmentací}. Program pomocí rotace a škálování přidá další upravené obrázky do datasetu.

Například:
```bash
python Main.py --preproces --class_augment '{"2": 300, "3": 100}'

python Main.py --preproces
```

### Trénování
Je nutné mít dataset o stejné struktuře a formátu souborů jako při předzpracování. Dataset musí být pojmenován jako `work_dataset`. Statistiky a stavy jednotlivých modelů se ukládájí do složky `UNet_results`, kterou program sám vytvoří.

Spuštění programu s argumenty: 
- **--train_model**: Flag pro trénování modelu.
- **--model_path**: Cesta k předtrénovanému modelu. Program jej načte a bude pokračovat v tréninku.
- **--classes_num**: Počet klasifikovatelných tříd. Program podle toho upraví výstup modelu.
- **--learning_rate**: Rychlost učení modelu.
- **--batch_size**: Počet vstupních vzorků obrázků, které model při tréniku zpracovává zároveň.
- **--num_epoch**: Počet epoch, kterými trénink modelu projde.
- **--pin_memory**: Použití pin memory u dataloaderu - může urychlit načítání dat.
- **--early_stop**: Počet epoch, po kterých se při nezlepšení dice skóre modelu trénování ukončí.
- **--class_weights**: List vah pro jednotlivé třídy.

Například:
```bash
python Main.py --train_model --num_epoch 10 --class_weights "[0.1, 2.5, 4, 6, 3.5]"

python Main.py --train_model --model_path "checkpoint.pth.tar" --classes_num 5 --learning_rate 0.0001 --batch_size 1 --num_epoch 50 --pin_memory True --early_stop 10 --class_weights "[0.05, 2.5, 2.5, 4.0, 2.0]"
```


### Uživatelské rozhraní
Spustí se rozhraní, kde lze vyzkoušet model na vlastních datech. Rozhraní slouží pro modely vytvořené tímto programem s architekturou UNet. Lze zde nahrát model a obrázek a následně pak vytvořit predikci modelu pro obrázek.

Spuštění programu s argumenty: 
- **--run_model**: Flag pro spuštění rozhraní k modelu.
- **--classes_num**: Počet klasifikovatelných tříd. Program podle toho upraví výstup modelu.

Například:
```bash
python Main.py --run_model

python Main.py --run_model --classes_num 2
```