import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Importiamo i nostri moduli personalizzati
from src import data_setup, models, engine

NUM_EPOCHS = 5  
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    print("\n--- Preparazione Dati (Fase 1: Binaria) ---")
    data_path = Path("data/raw")
    
    train_loader, val_loader, test_loader, classes = data_setup.create_dataloaders(
        data_dir=str(data_path), 
        phase='binary', 
        batch_size=BATCH_SIZE
    )
    
    # ECCO LA DIFFERENZA: Usiamo l'MLP di base!
    print("\n--- Inizializzazione Modello Baseline (MLP) ---")
    model = models.MLPBaseline(num_classes=1).to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n--- Inizio Addestramento MLP per {NUM_EPOCHS} epoche ---")
    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader, 
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )
    
    print("\n--- Salvataggio Modello ---")
    save_dir = Path("results/models")
    save_dir.mkdir(parents=True, exist_ok=True) 
    save_path = save_dir / "mlp_phase1_binary.pth"
    
    torch.save(model.state_dict(), save_path)
    print(f"Modello MLP salvato con successo in: {save_path}")

if __name__ == "__main__":
    main()
