
# 🧠 Classificazione Automatica del Grado di Demenza da Immagini MRI

**Progetto Accademico di Deep Learning — Università degli Studi di Catania (UniCT)**

👩‍💻 **Autori:** Massimiliano Cassia, Alessia Maccarrone, Martina Brancaforte

---

## 🎯 Obiettivo del Progetto

Lo scopo centrale di questo progetto non è la ricerca di architetture inedite, bensì la dimostrazione pratica e rigorosa della padronanza di un'intera pipeline di Deep Learning in produzione.

📄 **Documentazione Estesa:** Per un'analisi approfondita, consulta la Relazione Completa del Progetto (RelazioneDeepLearningCMM - NeuroVisionDL.pdf).


Partendo dal dataset **Augmented Alzheimer MRI Dataset** (immagini di risonanze magnetiche cerebrali), abbiamo ingegnerizzato un sistema in grado di classificare lo stato di avanzamento della demenza. Il progetto copre ogni fase: dall'Exploratory Data Analysis (EDA) alla Data Preparation (prevenendo il data leakage strutturale), passando per l'addestramento, il fine-tuning, un rigoroso benchmarking e, infine, il deploy tramite un'applicazione web interattiva.

---

## 🗄 Struttura della Repository

Il progetto è organizzato seguendo gli standard industriali per garantire riproducibilità e ordine:

```text
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

Per semplificare il dominio del problema e valutare le reali capacità estrattive delle reti, il task è stato diviso in due macrofasi di complessità crescente.

### Fase 1: Classificazione Binaria (Sano vs Affetto)

Le tre classi di malattia (Very Mild, Mild, Moderate) sono state accorpate in un'unica classe **"Affetto"** (Label 1), contrapposta a **"Sano"** (Label 0).

* **Modelli testati:** MLP (Baseline) e ResNet-18.
* **Obiettivo:** Valutare la capacità base dei modelli di distinguere un cervello sano da uno malato, stabilendo una *proof-of-concept* solida.
* **Loss Function:** `BCEWithLogitsLoss()` (Ottimizzata per output a singolo nodo).

### Fase 2: Classificazione Multiclasse (Le 4 Classi originali)

Le reti sono state sfidate a classificare l'esatto grado di demenza (0: Non Demented, 1: Very Mild, 2: Mild, 3: Moderate).

* **Modelli testati:** MLP (Baseline), MobileNetV2, ResNet-18 e ViT-B/16.
* **Obiettivo:** Spingere i modelli al limite per comprendere le loro difficoltà sulle classi di transizione (es. distinguere i sottili cambiamenti morfologici tra *Mild* e *Very Mild*).
* **Loss Function:** `CrossEntropyLoss()`.

---

## ⚙️ Data Engineering & Pipeline

Particolare attenzione è stata posta nella corretta gestione del flusso dati:

1. **Custom Dataset:** Implementazione di una classe `AlzheimerDataset` (ereditata da `torch.utils.data.Dataset`) per il re-mapping dinamico delle etichette "al volo" a seconda della fase (Binaria o Multiclasse), ottimizzando l'uso della memoria.
2. **Data Split Sicuro:** Utilizzo di `random_split` con seed fisso per la riproducibilità: **70% Train, 15% Validation, 15% Test**. Il test set è stato segregato ed escluso totalmente dalla fase di addestramento.
3. **Data Augmentation Sensata:** Sono state applicate le trasformazioni base (`Resize((224, 224))` e `ToTensor()`), seguite da una normalizzazione rigorosa basata su media e deviazione standard di ImageNet.

> *Nota tecnica:* Anche se le MRI appaiono in scala di grigi, la normalizzazione ImageNet sui 3 canali RGB è un requisito matematico obbligatorio per non sfalsare l'attivazione dei pesi durante il Transfer Learning.
> *Nota clinica:* Sono state volutamente escluse trasformazioni irrealistiche (es. il flip verticale), poiché anatomicamente un cervello non si presenta mai capovolto in una MRI.

---

## 🧠 Il "Model Zoo" (Architetture Neurali)

Tutti i modelli risiedono in `src/models.py` e sono stati gestiti tramite **Transfer Learning** (caricando i pesi pre-addestrati di ImageNet) e successivo **Fine-Tuning**.

1. **MLP (Baseline):** Una rete fully-connected "pura". Prende in input i pixel grezzi dell'immagine appiattiti in un vettore 1D di 150.528 valori, distruggendo volutamente la struttura spaziale. È usata come termine di paragone per dimostrare matematicamente l'inadeguatezza dei layer lineari semplici sulle immagini rispetto alle convoluzioni.
2. **MobileNetV2:** CNN leggera ed efficiente basata su *Depthwise Separable Convolutions*, studiata per contesti clinici con risorse computazionali limitate (es. tablet ospedalieri).
3. **ResNet-18:** CNN solida basata su connessioni residue (*skip connections*). Affidabile, performante e in grado di estrarre feature mediche complesse senza incappare nella degradazione del gradiente.
4. **ViT (Vision Transformer — `vit_b_16`):** Modello *State-of-the-Art* basato sul meccanismo di *Self-Attention* (analizza le immagini dividendole in "patch" 16x16). Utilizzato per testare i limiti hardware e comparare l'approccio classico (CNN) con quello dei Transformer.

---

## 🚀 Training Engine Avanzato

Il cuore pulsante del progetto (`src/engine.py`) supera il classico ciclo `for` accademico, implementando logiche tipiche degli ambienti di produzione:

* **Model Checkpointing:** Salvataggio dei pesi migliori (`.pth`) attivato esclusivamente in corrispondenza di un miglioramento della Validation Loss.
* **Early Stopping:** Interruzione forzata del training se la rete smette di apprendere per 5 epoche consecutive (*patience=5*), prevenendo attivamente l'overfitting.
* **Learning Rate Scheduler:** Utilizzo di `ReduceLROnPlateau` per far convergere dinamicamente il modello durante gli stalli, riducendo il tasso di apprendimento di un fattore 0.1.
* **Memory Management:** Gestione proattiva della VRAM per prevenire errori OOM (Out-Of-Memory) durante l'addestramento di modelli pesanti come il ViT.

---

## 📊 Benchmark e Risultati

L'accuratezza globale non è sufficiente per descrivere l'efficacia di un modello, specialmente in campo diagnostico. Per questo abbiamo basato la validazione su metriche combinate: **Precision, Recall, F1-Score (Macro)**, **Matrici di Confusione** e **Curve ROC/AUC**.

* **Fase 1 (Binaria):** Le reti convoluzionali hanno dominato il task. La ResNet-18 ha raggiunto prestazioni eccellenti (circa **98% di accuratezza**) con un Recall prossimo al **99%** per la classe "Affetto" (parametro clinico fondamentale per minimizzare i falsi negativi). L'MLP Baseline, come ampiamente previsto, si è fermato sotto l'**80%**, generando un inaccettabile tasso di falsi positivi.
* **Fase 2 (Multiclasse):** Dalle griglie 4x4 delle matrici di confusione è emerso visivamente il vero ostacolo clinico: le reti faticano a tracciare il confine esatto tra le classi di transizione morfologica (in particolare tra **Non Demented** e **Very Mild**).
* **Il caso ViT vs MobileNet:** Il Vision Transformer, pur essendo un modello all'avanguardia, si è rivelato computazionalmente proibitivo su hardware standard (es. GPU T4 di Colab), richiedendo un drastico ridimensionamento del Batch Size e restituendo performance globali non sempre superiori a quelle delle CNN classiche. Al contrario, la **MobileNetV2** si è aggiudicata il nostro benchmark interno, offrendo un F1-Score elevatissimo a fronte di un costo computazionale e tempi di inferenza minimi.

---

## 🌐 Produzione: La Web App Interattiva

Il ciclo di vita del progetto si conclude con la messa in produzione dei modelli addestrati. Tramite **Streamlit** (`src/app.py`), abbiamo sviluppato un'interfaccia grafica intuitiva (Decision Support System) che permette a un utente di:

1. Selezionare l'architettura desiderata dal menu laterale (es. testando la differenza di predizione tra ResNet e MobileNet).
2. Caricare una scansione MRI direttamente dal proprio dispositivo.
3. Ottenere un'inferenza in tempo reale sul tensore pre-processato.
4. Visualizzare la diagnosi finale supportata da una progress bar indicante le **probabilità (Softmax)** per ogni classe, essenziale per valutare la "confidenza" della rete e analizzare i casi limite.

---

## 💻 Istruzioni per l'uso (Come riprodurre il progetto)

**1. Clona la repository:**

```bash
git clone https://github.com/MaccarroneAlessia/unict-dlcmm-Maccarrone-Brancaforte-Cassia
cd alzheimer-mri-classification

```

**2. Installa le dipendenze:**

```bash
pip install -r requirements.txt

```

**3. Scarica il dataset:**
Inserisci la cartella decompressa dei dati grezzi scaricati da Kaggle all'interno del percorso `data/raw/`.

**4. Avvia l'addestramento:**

> ⚠️ **ATTENZIONE:** La cartella `results/models/` inizialmente è vuota. Per generare i file dei pesi addestrati (`.pth`) necessari per le inferenze dell'App Web, devi obbligatoriamente eseguire prima gli script di training. I file verranno salvati automaticamente nella directory corretta.

Per la **Fase 1 (Binaria):**

```bash
python train_phase1_mlp.py
python train_phase1.py

```

Per la **Fase 2 (Multiclasse):**

```bash
python train_phase2.py

```

**5. Avvia la Web App** *(solo dopo aver generato i file `.pth`)*:

```bash
streamlit run src/app.py

```

---

## ⚠️ Limitazioni e Assunzioni Cliniche (Data Leakage)

È doveroso segnalare una limitazione strutturale derivante dall'utilizzo dell'**Augmented Alzheimer MRI Dataset** di Kaggle. Poiché il dataset è stato pre-augmentato (bilanciato artificialmente) dall'autore originale offline, l'applicazione della funzione `random_split` di PyTorch (seppur con seed fisso) introduce un inevitabile grado di **Data Leakage**.

Questo accade perché varianti augmentate (es. ruotate o con contrasto modificato) della scansione dello stesso paziente potrebbero essere finite sia nel Train Set che nel Test Set, gonfiando artificialmente le metriche di accuratezza finali. In un contesto clinico reale, lo split dei dati deve avvenire rigorosamente a livello di paziente (**Patient-Level Split**) prima di applicare qualsiasi tecnica di augmentation. Ai fini di questo progetto accademico, le immagini sono state trattate come campioni indipendenti per potersi focalizzare sulla corretta ingegnerizzazione della pipeline MLOps.
