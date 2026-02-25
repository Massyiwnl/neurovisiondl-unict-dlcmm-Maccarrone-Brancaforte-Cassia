import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path

class AlzheimerDataset(Dataset):
    """
    Custom Dataset per gestire il re-mapping delle classi per la Fase 1 (Binaria) 
    e la Fase 2 (Multiclasse).
    """
    def __init__(self, root_dir, phase='multiclass', transform=None):
        # ImageFolder legge automaticamente le sottocartelle come classi
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.phase = phase
        
        # Mappatura originale di ImageFolder (alfabetica):
        # {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}
        self.class_to_idx = self.dataset.class_to_idx
        self.non_demented_idx = self.class_to_idx.get('NonDemented', 2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.phase == 'binary':
            # Se è NonDemented (Sano) -> classe 0
            # Se è qualsiasi altra forma di demenza (Affetto) -> classe 1
            label = 0 if label == self.non_demented_idx else 1
            
        return image, label

def create_dataloaders(data_dir: str, phase: str, batch_size: int = 32):
    """
    Crea i DataLoader per Train, Validation e Test (split 70% - 15% - 15%).
    """
    # Trasformazioni base: Resize standard per CNN/ViT, conversione in Tensore e Normalizzazione ImageNet
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Inizializziamo il nostro Custom Dataset
    full_dataset = AlzheimerDataset(root_dir=data_dir, phase=phase, transform=data_transforms)

    # Calcoliamo le dimensioni per lo split (70% train, 15% val, 15% test)
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Dividiamo il dataset usando un seed per la riproducibilità
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Creiamo i DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, full_dataset.class_to_idx

# Blocco di test rapido
if __name__ == "__main__":
    # Calcola il percorso assoluto in modo dinamico:
    # 1. Path(__file__).resolve().parent -> Trova la cartella 'src'
    # 2. .parent -> Sale alla root del progetto
    # 3. / "data" / "raw" -> Scende nella cartella corretta
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data" / "raw"
    
    if DATA_PATH.exists():
        print("Cartella trovata! Test Fase Binaria:")
        train_b, val_b, test_b, classes_b = create_dataloaders(str(DATA_PATH), phase='binary')
        print(f"Batch nel Train Loader (Binario): {len(train_b)}")
        
        print("\nTest Fase Multiclasse:")
        train_m, val_m, test_m, classes_m = create_dataloaders(str(DATA_PATH), phase='multiclass')
        print(f"Mappatura classi originali: {classes_m}")
    else:
        print(f"Cartella {DATA_PATH} non trovata.")
        print("Assicurati di aver estratto le cartelle del dataset dentro data/raw.")