import torch
from tqdm.auto import tqdm

class EarlyStopping:
    """
    Ferma il training se la validation loss non migliora dopo un certo numero di epoche (patience).
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"⚠️ Early Stopping counter: {self.counter} su {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               device: torch.device, is_binary: bool = False):
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        X, y = X.to(device), y.to(device)
        
        # Se usiamo BCEWithLogitsLoss per la binaria, le labels devono essere float
        if is_binary:
            y = y.float().unsqueeze(1)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcolo Accuracy
        if is_binary:
            predicted_classes = (torch.sigmoid(y_pred) > 0.5).float()
        else:
            predicted_classes = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            
        train_acc += (predicted_classes == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module, device: torch.device, is_binary: bool = False):
    
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validation", leave=False):
            X, y = X.to(device), y.to(device)
            
            if is_binary:
                y = y.float().unsqueeze(1)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

            if is_binary:
                predicted_classes = (torch.sigmoid(y_pred) > 0.5).float()
            else:
                predicted_classes = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                
            val_acc += (predicted_classes == y).sum().item() / len(y_pred)

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

def train_engine(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, 
                 val_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                 loss_fn: torch.nn.Module, epochs: int, device: torch.device, 
                 is_binary: bool = False, save_path: str = "best_model.pth", patience: int = 5):
    
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    early_stopping = EarlyStopping(patience=patience)
    
    # Inizializziamo il Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']}")
        
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device, is_binary)
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, device, is_binary)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # 1. Learning Rate Scheduler step
        scheduler.step(val_loss)
        
        # 2. Model Checkpointing: Salva i pesi solo se la Validation Loss migliora
        if val_loss < best_val_loss:
            print(f"⭐ Val Loss migliorata da {best_val_loss:.4f} a {val_loss:.4f}. Salvataggio modello...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
        # 3. Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping attivato all'epoca {epoch+1}. Il modello ha smesso di imparare.")
            break

    return results