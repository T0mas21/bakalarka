import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from PIL import Image

from User_interface.Model import ClassModel

'''
Třída pro frontend pro uživatelské rozhraní.
'''
class ClassView():
    def __init__(self, root, num_classes=5):
        self.root = root
        # Název okna
        self.root.title("Diabetic retinopathy detector")
        # Backend model (UNet)
        self.model = ClassModel()
        # Počet tříd segmentace
        self.num_classes = num_classes

        # Header bar 
        self.header_bar = tk.Frame(self.root, bg="#0d6efd", height=50)
        self.header_bar.pack(side=tk.TOP, fill=tk.X)

        self.header_label = tk.Label(
            self.header_bar,
            text="Nevybrán žádný model",
            bg="#0d6efd",
            fg="white",
            font=("Arial", 16, "bold"),
            pady=10
        )
        self.header_label.pack(side=tk.LEFT, padx=20)

        # Hlavní část
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left sidebar
        self.left_frame = tk.Frame(self.main_frame, width=150, bg="#add8e6")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.menu_label = tk.Label(self.left_frame, text="Menu", bg="#add8e6", font=("Arial", 14, "bold"))
        self.menu_label.pack(pady=(10, 20))

        self.btn_load_model = tk.Button(
            self.left_frame,
            text="Nahrát model",
            command=self.load_model,
            bg="#ffa500",
            activebackground="#ffb733"
        )
        self.btn_load_model.pack(pady=10, padx=10, fill=tk.X)

        self.btn_load_image = tk.Button(
            self.left_frame,
            text="Nahrát obrázek",
            command=self.load_image,
            bg="#ffa500",
            activebackground="#ffb733"
        )
        self.btn_load_image.pack(pady=10, padx=10, fill=tk.X)

        # Right sidebar
        self.right_frame = tk.Frame(self.main_frame, width=150, bg="#add8e6")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.right_label = tk.Label(self.right_frame, text="Akce", bg="#add8e6", font=("Arial", 14, "bold"))
        self.right_label.pack(pady=(10, 20))

        self.btn_run = tk.Button(
            self.right_frame,
            text="\u25B6️ Spustit model",
            command=self.run_model,
            anchor="w",
            padx=10
        )
        self.btn_run.pack(pady=10, padx=10, fill=tk.X)

        self.btn_clear_log = tk.Button(
            self.right_frame,
            text="Vymazat log",
            command=self.clear_log,
            anchor="w",
            padx=10
        )
        self.btn_clear_log.pack(pady=10, padx=10, fill=tk.X)

        # Center
        self.center_frame = tk.Frame(self.main_frame)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.original_frame = tk.Frame(self.center_frame, width=400, height=400)
        self.original_frame.pack_propagate(False)
        self.original_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.original_label = tk.Label(self.original_frame, text="Obrázek", bg="#ffffff")
        self.original_label.pack(expand=True, fill=tk.BOTH)

        self.prediction_frame = tk.Frame(self.center_frame, width=400, height=400)
        self.prediction_frame.pack_propagate(False)
        self.prediction_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.prediction_label = tk.Label(self.prediction_frame, text="Predikce", bg="#ffffff")
        self.prediction_label.pack(expand=True, fill=tk.BOTH)

        # Logovací terminál
        self.console_frame = tk.Frame(self.root, bg="#ffffff", highlightbackground="gray", highlightthickness=3)
        self.console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH)
        self.console_frame.configure(height=250)
        self.console_frame.pack_propagate(False)

        self.log_label = tk.Label(
            self.console_frame,
            text="Log:",
            bg="#ffffff",
            fg="black",
            font=("Arial", 12, "bold"),
            anchor="w"
        )
        self.log_label.pack(side=tk.TOP, anchor="w", padx=5, pady=(5, 0))

        self.console_text = tk.Text(
            self.console_frame,
            bg="#ffffff",
            fg="black",
            insertbackground="black",
            borderwidth=0,
            highlightthickness=0
        )
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.console_scrollbar = tk.Scrollbar(self.console_frame, command=self.console_text.yview)
        self.console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.config(yscrollcommand=self.console_scrollbar.set)

        self.original_label.bind("<Configure>", lambda e: self.update_image(self.original_image, self.original_label) if hasattr(self, 'original_image') else None)
        self.prediction_label.bind("<Configure>", lambda e: self.update_image(self.prediction_image, self.prediction_label) if hasattr(self, 'prediction_image') else None)

    # Logování zpráv
    def log(self, message):
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)

    # Spuštění view
    def start(self):
        self.root.mainloop()

    # Vymazání logu
    def clear_log(self):
        self.console_text.delete('1.0', tk.END)
    
    # Načtení modelu
    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth;*.pt;*.pth.tar")])
        if file_path:
            self.model.set_model(model_path=file_path, classes_num=self.num_classes)
            file_name = os.path.basename(file_path)
            self.header_label.config(text=f"Model: {file_name}")
            self.log(f"✅ Model '{file_name}' úspěšně načten.")

    # Načtení obrázku
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            self.model.set_image(file_path)
            self.original_image = Image.open(file_path)
            self.update_image(self.original_image, self.original_label)
            self.log(f"✅ Obrázek '{os.path.basename(file_path)}' úspěšně načten.")

    # Spuštění predikce
    def run_model(self):
        result = self.model.get_results()
        if result is not None:
            self.prediction_image = Image.fromarray(result)
            self.update_image(self.prediction_image, self.prediction_label)
            self.log("✅ Predikce byla úspěšně provedena a zobrazena.")
        else:
            self.log("❌ Chyba: Model nebo obrázek nebyl správně načten.")

    def update_image(self, pil_image, label_widget):
        # Získání velikosti widgetu
        label_widget.update_idletasks()
        w = label_widget.winfo_width()
        h = label_widget.winfo_height()
        
        # Úprava velikosti podle obrazovky
        if w > 1 and h > 1:
            resized_image = pil_image.copy().resize((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            label_widget.config(image=photo)
            label_widget.image = photo