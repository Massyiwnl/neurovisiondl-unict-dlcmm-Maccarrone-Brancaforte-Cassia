# --- FIX SYMPY ---
import sympy
import sympy.printing
# -----------------

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Setup Path dinamico: se l'app è in src/, ".." punterà alla root corretta
possible_roots = ["../", "./", "/content/neurovisiondl-unict-dlcmm-Maccarrone-Brancaforte-Cassia/"]
PROJECT_ROOT = None
for path in possible_roots:
    if os.path.exists(os.path.join(path, "src")):
        PROJECT_ROOT = path
        break
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.models import MLPBaseline, get_mobilenet_v2, get_resnet18, get_vit_b_16
except Exception as e:
    st.error(f"Errore critico di importazione moduli: {e}")
    st.stop()

st.set_page_config(page_title="NeuroVision MRI", page_icon="🧠", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

@st.cache_resource
def load_model(model_name):
    MODELS_DIR = os.path.join(PROJECT_ROOT, "results/models/") if PROJECT_ROOT else "results/models/"
    
    if model_name == "MLP Baseline":
        model = MLPBaseline(num_classes=4)
        file_name = "mlp_baseline_phase2.pth"
    elif model_name == "MobileNetV2":
        model = get_mobilenet_v2(num_classes=4, pretrained=False)
        file_name = "mobilenetv2_phase2.pth"
    elif model_name == "ResNet18":
        model = get_resnet18(num_classes=4, pretrained=False)
        file_name = "resnet18_phase2.pth"
    else:  
        model = get_vit_b_16(num_classes=4, pretrained=False)
        file_name = "vit_b_16_phase2.pth"
        
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        return None, path
        
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model, path

def process_image(image):
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# --- UI ---
st.title("🧠 NeuroVision Demo")
st.markdown("Carica una risonanza magnetica (MRI) e utilizza i modelli per prevedere lo stadio della demenza.")

st.sidebar.header("⚙️ Impostazioni")
selected_model_name = st.sidebar.selectbox("Scegli l'architettura neurale:", ("ResNet18", "MobileNetV2", "ViT_B_16", "MLP Baseline"))
st.sidebar.info(f"💻 Elaborazione in corso su: **{device.type.upper()}**")

uploaded_file = st.file_uploader("Carica un'immagine MRI (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="MRI Caricata", use_container_width=True)
        
    with col2:
        with st.spinner(f"Caricamento {selected_model_name}..."):
            model, path = load_model(selected_model_name)
            
        if model is None:
            st.error(f"⚠️ Pesi non trovati in {path}.")
        else:
            with st.spinner("Estrazione predizione..."):
                input_tensor = process_image(image)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                    predicted_class_idx = probabilities.argmax()
                    predicted_class_name = CLASSES[predicted_class_idx]
                    confidence = probabilities[predicted_class_idx] * 100

            st.subheader("Risultato")
            if predicted_class_idx == 2: 
                st.success(f"**{predicted_class_name}** ({confidence:.1f}%)")
            else:
                st.error(f"**{predicted_class_name}** ({confidence:.1f}%)")
                
            st.markdown("### Probabilità:")
            for i, class_name in enumerate(CLASSES):
                st.write(f"{class_name}: {probabilities[i]*100:.1f}%")
                st.progress(float(probabilities[i]))