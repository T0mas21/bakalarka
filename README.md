# Program pro segmentaci příznaků diabetické retinopatie
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
- **--class_augment**: Musí k sobě mít parametr slovník {hodnota třídy : maximální počet augmentací}. Program pomocí rotace a škálování přidá další upravené obrázky do datasetu.
- **--treshold_high**: Musí k sobě mít číselnou hodnotu, která značí hranici, kdy se pixely nad touto hodnotou nastaví na 255.
- **--treshold_low**: Musí k sobě mít číselnou hodnotu, která značí hranici, kdy se pixely pod touto hodnotou nastaví na 0.
- **--split**: Musí k sobě mít hodnotu True nebo False. Pokud je True, tak rozdělí předzpracované obrázky na 4 části. 

Například:
```bash
python Main.py --preproces --class_augment '{"2": 300, "3": 100}' --treshold_high 200 --treshold_low 100

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
- **--nested_unet**: Využití rozšířené architektury UNet.

Například:
```bash
python Main.py --train_model --num_epoch 10 --class_weights "[0.1, 2.5, 4, 6, 3.5]"

python Main.py --train_model --model_path "checkpoint.pth.tar" --classes_num 5 --learning_rate 0.0001 --batch_size 1 --num_epoch 50 --pin_memory True --early_stop 10 --class_weights "[0.1, 2.0, 3.0, 4.0, 2.0]" --nested_unet True
```

### Testování
Program otestuje výstupy modelu na validačním datasetu a na výstup vypíše hodnoty metrik. Pro otestování je nutné mít validační dataset stejný jako při trénování. Cesta ke složce s validačním datasetem musí být ve formátu `work_dataset/validate`.

Spuštění programu s argumenty: 
- **--test_model**: Flag pro testování modelu.
- **--model_path**: Cesta k testovanému modelu.
- **--classes_num**: Počet klasifikovatelných tříd.
- **--batch_size**: Počet vstupních vzorků obrázků, které model při tréniku zpracovává zároveň.
- **--pin_memory**: Použití pin memory u dataloaderu - může urychlit načítání dat.
- **--nested_unet**: Využití rozšířené architektury UNet.

```bash
python Main.py --test_model --model_path "checkpoint.pth.tar"

python Main.py --test_model --model_path "checkpoint.pth.tar" --classes_num 5 --batch_size 1 --pin_memory True --nested_unet True
```


### Uživatelské rozhraní
Spustí se rozhraní, kde lze vyzkoušet model na vlastních datech. Rozhraní slouží pro modely vytvořené tímto programem s architekturou UNet. Lze zde nahrát model a obrázek a následně pak vytvořit predikci modelu pro obrázek. Obrázek je po nahrání zpracován, aby byla segmentace efektivnější, proto je i možné manuálně nastavit hranice prahování.

Spuštění programu s argumenty: 
- **--run_model**: Flag pro spuštění rozhraní k modelu.
- **--classes_num**: Počet klasifikovatelných tříd. Program podle toho upraví výstup modelu.
- **--treshold_high**: Musí k sobě mít číselnou hodnotu, která značí hranici, kdy se pixely nad touto hodnotou nastaví na 255. Mělo by se nastavit na modely trénované s prahováním.
- **--treshold_low**: Musí k sobě mít číselnou hodnotu, která značí hranici, kdy se pixely pod touto hodnotou nastaví na 0. Mělo by se nastavit na modely trénované s prahováním.
- **--nested_unet**: Využití rozšířené architektury UNet.

Například:
```bash
python Main.py --run_model

python Main.py --run_model --classes_num 2 --nested_unet True
```

### Natrénované modely
Ve složce `Pretrained_models` nebo na google disku [natrénované modely](https://drive.google.com/drive/folders/1qyvVL8UBRe3M037B5ZNVmL69lQMBvRad?usp=sharing), se nacházejí předtrénované modely pro segmentaci onemocnění diabetické retinopatie. Soubory jsou pojmenované jako `<architektura>_<třída>*.pth.tar`, kde architektura je v tomto případě UNet, třídy můžou být EX - tvrdé exsudáty, SE - měkké exsudáty, MA - mikroaneurysmata, HE - hemoragie nebo ALL - všechny předchozí třídy.