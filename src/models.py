import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. Modello Baseline: Multi-Layer Perceptron (MLP)
# ==========================================
class MLPBaseline(nn.Module):
    """
    Modello MLP di base. Poiché le immagini sono 224x224 a 3 canali (RGB),
    l'input appiattito sarà molto grande: 224 * 224 * 3 = 150528.
    Serve solo per dimostrare le scarse performance rispetto alle CNN.
    """
    def __init__(self, num_classes=1):
        super(MLPBaseline, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(224 * 224 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Aiuta a prevenire l'overfitting immediato
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# ==========================================
# 2. Modello CNN Leggero: MobileNetV2
# ==========================================
def get_mobilenet_v2(num_classes=1, pretrained=True):
    """
    Carica MobileNetV2 pre-addestrata su ImageNet e adatta l'ultimo layer.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    # Sostituiamo il classificatore finale
    # MobileNetV2 ha il classificatore in model.classifier[1]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

# ==========================================
# 3. Modello CNN Classico: ResNet18
# ==========================================

def get_resnet18(num_classes=1, pretrained=True):
    """
    Carica ResNet18 pre-addestrata su ImageNet. Ottimo bilanciamento tra 
    performance e costo computazionale.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Sostituiamo l'ultimo layer fully connected (fc)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# ==========================================
# 4. Modello State-of-the-Art: Vision Transformer (ViT)
# ==========================================
def get_vit_b_16(num_classes=1, pretrained=True):
    """
    Carica Vision Transformer (Base, patch size 16). 
    Attenzione: richiede più VRAM rispetto alle CNN.
    """
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    
    # ViT ha la testa di classificazione in model.heads.head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model

# Blocco di test rapido per verificare le architetture
if __name__ == "__main__":
    print("Testiamo l'inizializzazione dei modelli per la Fase 2 (Multiclasse)...")
    
    # Creiamo un tensore fittizio che simula un batch di 2 immagini 224x224 RGB
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test MLP
    mlp = MLPBaseline(num_classes=4)
    print(f"Output MLP shape: {mlp(dummy_input).shape} -> Atteso: [2, 4]")
    
    # Test ResNet18
    resnet = get_resnet18(num_classes=4)
    print(f"Output ResNet18 shape: {resnet(dummy_input).shape} -> Atteso: [2, 4]")
    
    print("Modelli configurati correttamente!")