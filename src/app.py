import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Assicuriamoci che la cartella 'src' sia nel path
sys.path.append(os.path.abspath("."))
from src.models import MLPBaseline, get_mobilenet_v2, get_resnet18, get_vit_b_16

# ==========================================
# CONFIGURAZIONE PAGINA E VARIABILI
# ==========================================
st.set_page_config(page_title="NeuroVision MRI | Alzheimer Detection", page_icon="🧠", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dizionario esatto delle classi (come da dataloader Fase 2)
CLASSES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# ==========================================
# FUNZIONI DI SUPPORTO (Con Cache per velocità)
# ==========================================
@st.cache_resource
def load_model(model_name):
    """Carica il modello selezionato e i suoi pesi, tenendoli in cache nella RAM."""
    MODELS_DIR = "results/models/"
    
    if model_name == "MLP Baseline":
        model = MLPBaseline(num_classes=4)
        file_name = "mlp_baseline_phase2.pth"
    elif model_name == "MobileNetV2":
        model = get_mobilenet_v2(num_classes=4, pretrained=False)
        file_name = "mobilenetv2_phase2.pth"
    elif model_name == "ResNet18":
        model = get_resnet18(num_classes=4, pretrained=False)
        file_name = "resnet18_phase2.pth"
    else:  # ViT_B_16
        model = get_vit_b_16(num_classes=4, pretrained=False)
        file_name = "vit_b_16_phase2.pth"
        
    path = os.path.join(MODELS_DIR, file_name)
    
    if not os.path.exists(path):
        st.error(f"⚠️ Pesi non trovati per {model_name} in {path}. Hai addestrato questo modello?")
        return None
        
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

def process_image(image):
    """Applica le stesse trasformazioni usate in addestramento."""
    # Convertiamo in RGB in caso di immagini in scala di grigi a 1 canale
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device) # Aggiunge la dimensione del batch

# ==========================================
# INTERFACCIA UTENTE (UI)
# ==========================================
st.title("🧠 NeuroVision: Alzheimer MRI Classifier")
st.markdown("""
Questa demo interattiva permette di caricare una risonanza magnetica (MRI) e utilizzare i modelli di Deep Learning addestrati per prevedere lo stadio della demenza di Alzheimer.
""")

st.sidebar.header("⚙️ Impostazioni")
selected_model_name = st.sidebar.selectbox(
    "Scegli l'architettura neurale:",
    ("ResNet18", "MobileNetV2", "ViT_B_16", "MLP Baseline")
)

st.sidebar.info(f"💻 Elaborazione in corso su: **{device.type.upper()}**")

# Caricamento Modello
model = load_model(selected_model_name)

# Upload Immagine
uploaded_file = st.file_uploader("Carica un'immagine MRI (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Mostra l'immagine
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="MRI Caricata", use_container_width=True)
        
    # 2. Elaborazione e Predizione
    with st.spinner(f"Analisi in corso con {selected_model_name}..."):
        input_tensor = process_image(image)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            predicted_class_idx = probabilities.argmax()
            predicted_class_name = CLASSES[predicted_class_idx]
            confidence = probabilities[predicted_class_idx] * 100

    # 3. Mostra i Risultati
    with col2:
        st.subheader("Risultato dell'Analisi")
        
        # Colore dinamico in base alla classe (Verde per Sano, Rosso/Arancio per Affetto)
        if predicted_class_idx == 2: # Non Demented
            st.success(f"**{predicted_class_name}** ({confidence:.1f}%)")
        else:
            st.error(f"**{predicted_class_name}** ({confidence:.1f}%)")
            
        st.markdown("### Probabilità per classe:")
        # Creiamo delle barre di progresso per ogni probabilità
        for i, class_name in enumerate(CLASSES):
            st.write(f"{class_name}: {probabilities[i]*100:.1f}%")
            st.progress(float(probabilities[i]))