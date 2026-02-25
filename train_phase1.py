import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src import data_setup, models, engine

# Parametri globali vanno bene qui fuori
NUM_EPOCHS = 5  
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def main():
    # SPOSTATO QUI DENTRO!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    print("\n--- Preparazione Dati (Fase 1: Binaria) ---")
    data_path = Path("data/raw")
    
    train_loader, val_loader, test_loader, classes = data_setup.create_dataloaders(
        data_dir=str(data_path), 
        phase='binary', 
        batch_size=BATCH_SIZE
    )
    print(f"Batch di training: {len(train_loader)}")
    
    # 4. Inizializzazione del Modello
    print("\n--- Inizializzazione Modello (ResNet18) ---")
    # num_classes=1 perché è una classificazione binaria (Sano = 0, Affetto = 1)
    model = models.get_resnet18(num_classes=1, pretrained=True).to(device)
    
    # 5. Definizione della Loss e dell'Ottimizzatore
    # BCEWithLogitsLoss è la standard per la classificazione binaria in PyTorch
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Avvio dell'Addestramento
    print(f"\n--- Inizio Addestramento per {NUM_EPOCHS} epoche ---")
    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader, # Usiamo il validation set per monitorare l'overfitting
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )
    
    # 7. Salvataggio del modello addestrato
    print("\n--- Salvataggio Modello ---")
    save_dir = Path("results/models")
    save_dir.mkdir(parents=True, exist_ok=True) # Crea la cartella se non esiste
    save_path = save_dir / "resnet18_phase1_binary.pth"
    
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato con successo in: {save_path}")

if __name__ == "__main__":
    main()