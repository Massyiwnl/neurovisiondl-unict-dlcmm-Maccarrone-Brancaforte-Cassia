import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Esegue un'epoca di addestramento."""
    # Mette il modello in modalità training (attiva Dropout, BatchNorm, ecc.)
    model.train()
    
    train_loss, train_acc = 0, 0
    
    # Loop sui batch
    for batch, (X, y) in enumerate(dataloader):
        # Sposta i dati sul dispositivo (GPU/CPU)
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (Calcolo delle predizioni)
        # Se usiamo la classificazione binaria con BCEWithLogitsLoss, le label devono essere float
        if loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
            y = y.unsqueeze(1).float()
            
        y_pred = model(X)

        # 2. Calcolo della Loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad (Pulisce i gradienti accumulati)
        optimizer.zero_grad()

        # 4. Loss backward (Backpropagation)
        loss.backward()

        # 5. Optimizer step (Aggiornamento dei pesi)
        optimizer.step()

        # Calcolo dell'accuratezza (diverso tra binario e multiclasse)
        if loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
            y_pred_class = torch.round(torch.sigmoid(y_pred))
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        else:
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Calcola le medie per l'intera epoca
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Valuta il modello sui dati di Validation/Test."""
    # Mette il modello in modalità valutazione (disattiva Dropout, BatchNorm, ecc.)
    model.eval() 
    
    test_loss, test_acc = 0, 0
    
    # Disattiva il calcolo dei gradienti per risparmiare memoria e tempo
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            if loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
                y = y.unsqueeze(1).float()

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calcolo della loss e dell'accuratezza
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            if loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
                y_pred_class = torch.round(torch.sigmoid(y_pred))
                test_acc += (y_pred_class == y).sum().item() / len(y_pred)
            else:
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                test_acc += (y_pred_class == y).sum().item() / len(y_pred)
                
    # Calcola le medie
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    """Unisce train_step e test_step ed esegue il loop per N epoche."""
    
    # Dizionario per salvare i risultati (utile poi per i plot)
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # Stampa i risultati dell'epoca
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {test_loss:.4f} | val_acc: {test_acc:.4f}")
        
        # Salvataggio dei risultati nel dizionario
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results