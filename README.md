# 🧠 NEUROVISION DL - Classificazione Automatica del Grado di Demenza da Immagini MRI

**Progetto Accademico di Deep Learning — Università degli Studi di Catania (UniCT)**

👩‍💻 **Autori:** Massimiliano Cassia, Alessia Maccarrone, Martina Brancaforte

---

## 🎯 Obiettivo del Progetto

Lo scopo centrale di questo progetto non è la ricerca di architetture inedite, bensì la dimostrazione pratica e rigorosa della padronanza di un'intera pipeline di Deep Learning in produzione.

Partendo dal dataset **Augmented Alzheimer MRI Dataset** (immagini di risonanze magnetiche cerebrali), abbiamo ingegnerizzato un sistema in grado di classificare lo stato di avanzamento della demenza. Il progetto copre ogni fase: dall'Exploratory Data Analysis (EDA) alla Data Preparation (prevenendo il data leakage), passando per l'addestramento, il fine-tuning, un rigoroso benchmarking e, infine, il deploy tramite un'applicazione web interattiva.

---

## 🗄 Struttura della Repository

Il progetto è organizzato seguendo gli standard industriali per garantire riproducibilità e ordine:

```
alzheimer-mri-classification/
│
├── data/                             # (Ignorata in .gitignore)
│   └── raw/                          # Dataset originale scaricato da Kaggle
│
├── notebooks/                        # Jupyter Notebooks per esplorazione e analisi visiva
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis e bilanciamento classi
│   ├── 02_benchmark.ipynb            # Benchmark e visualizzazione Fase 1 (Binaria)
│   └── 03_benchmark_multiclass.ipynb # Benchmark e visualizzazione Fase 2
│
├── src/                              # Codice sorgente core (.py)
│   ├── data_setup.py                 # Custom Dataset, DataLoader e data split
│   ├── engine.py                     # Training & Validation loop professionali
│   ├── models.py                     # Definizione e importazione architetture (Transfer Learning)
│   ├── utils.py                      # Funzioni di supporto (plot, salvataggio)
│   └── app.py                        # Script Streamlit per l'interfaccia Web
│
├── results/                          # Output generati (grafici, metriche)
│   └── models/                       # Cartella destinata ai pesi salvati (.pth)
│
├── train_phase1.py                   # Eseguibile: Training Fase 1 (Binaria)
├── mlp_train_phase1.py               # Eseguibile: Training Baseline MLP
├── train_phase2.py                   # Eseguibile: Training Fase 2 (Multiclasse)
├── requirements.txt                  # Dipendenze del progetto
└── README.md                         # Documentazione (Questo file)
```

---

## 🔬 Metodologia e Strategia (Le Due Fasi)

Per semplificare il dominio del problema e valutare le reali capacità estrattive delle reti, il task è stato diviso in due macrofasi.

### Fase 1: Classificazione Binaria (Sano vs Affetto)

Le tre classi di malattia (Very Mild, Mild, Moderate) sono state accorpate in un'unica classe **"Affetto"** (Label 1), contrapposta a **"Sano"** (Label 0).

- **Obiettivo:** Valutare la capacità base dei modelli di distinguere un cervello sano da uno malato.
- **Loss Function:** `BCEWithLogitsLoss()`

### Fase 2: Classificazione Multiclasse (Le 4 Classi originali)

Le reti sono state sfidate a classificare l'esatto grado di demenza (0, 1, 2, 3).

- **Obiettivo:** Comprendere le difficoltà del modello (es. la sottile differenza tra Mild e Very Mild).
- **Loss Function:** `CrossEntropyLoss()`

---

## ⚙️ Data Engineering & Pipeline

Particolare attenzione è stata posta nella prevenzione del **Data Leakage**:

1. **Custom Dataset:** Implementazione di una classe `AlzheimerDataset` (ereditata da `torch.utils.data.Dataset`) per il re-mapping dinamico delle etichette a seconda della fase (Binaria o Multiclasse).
2. **Data Split Sicuro:** Utilizzo di `random_split` con seed fisso: **70% Train, 15% Validation, 15% Test**. Il test set è stato segregato ed escluso totalmente dalla fase di addestramento.
3. **Data Augmentation Sensata:** Sono state applicate `Resize((224, 224))`, conversione in Tensore e normalizzazione (mean, std) di ImageNet. *Nota clinica: Sono state volutamente escluse trasformazioni irrealistiche come il flip verticale (un cervello non si presenta mai sottosopra in una MRI).*

---

## 🧠 Il "Model Zoo" (Architetture Neurali)

Tutti i modelli risiedono in `src/models.py` e sono stati gestiti tramite **Transfer Learning** (pesi ImageNet) e **Fine-Tuning**.

1. **MLP (Baseline):** Una rete fully-connected di base usata come termine di paragone per dimostrare matematicamente l'inadeguatezza delle reti lineari sulle immagini rispetto alle CNN.
2. **MobileNetV2:** CNN leggera ed efficiente, studiata per contesti con risorse computazionali limitate.
3. **ResNet18:** CNN basata su connessioni residue. Affidabile, performante e in grado di estrarre feature complesse.
4. **ViT (Vision Transformer — `vit_b_16`):** Modello State-of-the-Art basato sull'attenzione (analizza le immagini per "patch"). Utilizzato per testare i limiti hardware e comparare l'approccio classico CNN con quello basato su transformer.

---

## 🚀 Training Engine Avanzato

Il cuore pulsante del progetto (`src/engine.py`) supera il classico ciclo `for` scolastico implementando tecniche da produzione:

- **Model Checkpointing:** Salvataggio dei pesi (`.pth`) solo al miglioramento della Validation Loss.
- **Early Stopping:** Interruzione del training se la rete smette di apprendere per 5 epoche, prevenendo attivamente l'overfitting.
- **Learning Rate Scheduler:** Utilizzo di `ReduceLROnPlateau` per far convergere dinamicamente il modello durante gli stalli.
- **DevOps & RAM Management:** Implementazione di `torch.cuda.empty_cache()` per prevenire errori OOM (Out-Of-Memory) durante l'addestramento sequenziale dei modelli, unito a strategie di ripristino per gestire le disconnessioni tipiche di Google Colab.

---

## 📊 Benchmark e Risultati

L'accuratezza globale non è sufficiente, specialmente in campo medico. Per questo abbiamo basato la validazione su: **Precision, Recall, F1-Score (Macro)**, **Matrici di Confusione** e **Curve ROC/AUC**.

- **Fase 1 (Binaria):** Le reti convoluzionali hanno dominato. ResNet18 ha raggiunto quasi il **98% di accuratezza** con un Recall del **99%** per la classe "Affetto" (fondamentale in campo medico per evitare i falsi negativi). L'MLP, come previsto, si è fermato al **79%**, generando un alto tasso di falsi positivi.
- **Fase 2 (Multiclasse):** Dalle griglie 4×4 delle matrici di confusione è emerso visivamente l'ostacolo clinico: le reti faticano a tracciare il confine esatto tra le classi di transizione (**Mild vs Very Mild**). Le curve ROC One-vs-Rest (OvR) hanno confermato queste discrepanze.
- **Il caso ViT:** Pur essendo un modello all'avanguardia, la sua voracità computazionale non ha giustificato l'aumento di performance rispetto a ResNet18 su un dataset di queste dimensioni, confermando l'importanza della scelta del modello in base al contesto.

---

## 🌐 Produzione: La Web App Interattiva

Il ciclo di vita del progetto si conclude con la messa in produzione del modello. Tramite **Streamlit** (`src/app.py`), abbiamo sviluppato un'interfaccia grafica intuitiva che permette a un utente di:

1. Selezionare l'architettura desiderata dal menu laterale.
2. Caricare un'immagine MRI dal proprio dispositivo.
3. Ottenere un'inferenza in tempo reale sul tensore pre-processato.
4. Visualizzare la diagnosi con colori dinamici (**Verde = Sano**, **Rosso = Affetto**) e una progress bar indicante le probabilità (Softmax) per ogni classe.

Il deploy è stato gestito dinamicamente da Google Colab esponendo la porta locale tramite tunnel **Cloudflare**.

---

## 💻 Istruzioni per l'uso (Come riprodurre il progetto)

**1. Clona la repository:**

```bash
git clone https://github.com/tuo-username/alzheimer-mri-classification.git
cd alzheimer-mri-classification
```

**2. Installa le dipendenze:**

```bash
pip install -r requirements.txt
```

**3. Scarica il dataset:**

Inserisci i file di Kaggle all'interno della cartella `data/raw/`.

**4. Avvia l'addestramento:**

> ⚠️ **ATTENZIONE IMPORTANTE SUI FILE DEI MODELLI:** La cartella `results/models/` inizialmente è vuota. Per generare i file dei pesi addestrati (`.pth`) necessari per le inferenze e l'App, devi obbligatoriamente eseguire gli script di training. I file `.pth` verranno generati e salvati automaticamente nella cartella corretta al termine di questi script.

Per la **Fase 1 (Binaria):**

```bash
python train_phase1.py
python mlp_train_phase1.py
```

Per la **Fase 2 (Multiclasse):**

```bash
python train_phase2.py
```

**5. Avvia la Web App** *(solo dopo aver generato i file `.pth`)*:

```bash
streamlit run src/app.py
```
