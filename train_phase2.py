import os
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from pathlib import Path

# Importiamo le tue funzioni custom
from src.data_setup import create_dataloaders
from src.engine import train_engine
from src.models import MLPBaseline, get_mobilenet_v2, get_resnet18, get_vit_b_16

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
BATCH_SIZE = 16 # Ridotto a 16 per non far crashare la GPU su Colab col ViT
EPOCHS = 20     # Mettiamo 20 epoche, tanto l'Early Stopping fermerà prima se necessario
LEARNING_RATE = 1e-4 # LR basso per il fine-tuning
NUM_CLASSES = 4 # Fase 2: 4 classi separate
PATIENCE = 5    # Dopo 5 epoche senza miglioramenti, l'Early Stopping interviene

# Creiamo la cartella per salvare i pesi se non esiste
RESULTS_DIR = Path("results/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device in uso: {device}")

# ==========================================
# 1. CARICAMENTO DATI (MULTICLASSE)
# ==========================================
DATA_DIR = Path("data/raw")
if not DATA_DIR.exists():
    raise FileNotFoundError("⚠️ Cartella data/raw non trovata. Assicurati che i dati siano presenti.")

print("\n📦 Caricamento Dataloaders in modalità MULTICLASSE...")
train_loader, val_loader, test_loader, classes = create_dataloaders(
    data_dir=str(DATA_DIR), 
    phase='multiclass', 
    batch_size=BATCH_SIZE
)
print(f"Classi trovate: {classes}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# Loss function per la classificazione multiclasse
loss_fn = nn.CrossEntropyLoss()

# ==========================================
# 2. DEFINIZIONE DEL MODEL ZOO
# ==========================================
# Creiamo un dizionario con i modelli da addestrare.
# In questo modo lo script li elaborerà in sequenza in un solo colpo.
models_to_train = {
    "MLP_Baseline": MLPBaseline(num_classes=NUM_CLASSES),
    "MobileNetV2": get_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True),
    "ResNet18": get_resnet18(num_classes=NUM_CLASSES, pretrained=True),
    "ViT_B_16": get_vit_b_16(num_classes=NUM_CLASSES, pretrained=True)
}

# ==========================================
# 3. LOOP DI ADDESTRAMENTO PER OGNI MODELLO
# ==========================================
for model_name, model in models_to_train.items():
    print(f"\n{'='*50}")
    print(f"🚀 INIZIO TRAINING: {model_name}")
    print(f"{'='*50}")
    
    # Spostiamo il modello sulla GPU
    model = model.to(device)
    
    # Optimizer (Adam è ottimo per transfer learning)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Definiamo il percorso di salvataggio per questo modello
    save_path = str(RESULTS_DIR / f"{model_name.lower()}_phase2.pth")
    
    # Lanciamo il training usando il nuovo engine
    # is_binary=False perché siamo in multiclasse (usa Softmax e Argmax)
    results = train_engine(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device,
        is_binary=False,
        save_path=save_path,
        patience=PATIENCE
    )
    
    print(f"✅ Training completato per {model_name}. Migliori pesi salvati in {save_path}")
    
    # --- PULIZIA MEMORIA ---
    # Fondamentale quando si addestrano più modelli in sequenza sulla stessa GPU
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

print("\n🎉 TUTTE LE FASI DI TRAINING SONO COMPLETATE!")
print("Controlla la cartella results/models/ per i file .pth")