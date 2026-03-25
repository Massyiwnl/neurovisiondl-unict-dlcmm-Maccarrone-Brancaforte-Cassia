# NeuroVision DL — Classificazione Automatica della Demenza da Immagini MRI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Licenza](https://img.shields.io/badge/Licenza-Accademica-blue?style=flat-square)
![Stato](https://img.shields.io/badge/Stato-Completato-success?style=flat-square)

**Progetto Accademico di Deep Learning — Università degli Studi di Catania (UniCT)**

*Massimiliano Cassia · Alessia Maccarrone · Martina Brancaforte*

[Panoramica](#-panoramica) · [Metodologia](#-metodologia) · [Risultati](#-risultati) · [Installazione](#-installazione) · [Utilizzo](#-utilizzo) · [Limitazioni](#-limitazioni)

</div>

---

## Panoramica

Questo progetto implementa una pipeline di deep learning end-to-end per la **classificazione automatica del grado di avanzamento della malattia di Alzheimer** a partire da immagini di risonanza magnetica cerebrale. L'obiettivo principale non è la ricerca di architetture inedite, bensì la dimostrazione pratica e rigorosa della padronanza di un'intera pipeline MLOps in produzione — dall'analisi esplorativa dei dati fino al deploy di un'applicazione web interattiva.

Il sistema classifica la gravità della demenza usando l'[Augmented Alzheimer MRI Dataset](https://www.kaggle.com/), articolato in due task di complessità crescente:

| Fase | Task | Classi | Architetture |
|---|---|---|---|
| Fase 1 | Classificazione Binaria | Sano vs. Affetto | MLP (baseline), ResNet-18 |
| Fase 2 | Classificazione Multiclasse | Non / Very Mild / Mild / Moderate Demented | MLP, MobileNetV2, ResNet-18, ViT-B/16 |

> 📄 Per un'analisi approfondita dell'intero progetto, consulta la relazione tecnica: `RelazioneDeepLearningCMM - NeuroVisionDL.pdf`.

---

## Struttura della Repository

```
alzheimer-mri-classification/
│
├── data/
│   └── raw/                          # Dataset originale Kaggle (escluso dal versioning)
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Analisi Esplorativa dei Dati e bilanciamento classi
│   ├── 02_benchmark.ipynb            # Benchmark e visualizzazioni Fase 1 (Binaria)
│   └── 03_benchmark_multiclass.ipynb # Benchmark e visualizzazioni Fase 2 (Multiclasse)
│
├── src/
│   ├── data_setup.py                 # Custom Dataset, DataLoader e split stratificato
│   ├── engine.py                     # Loop di training e validazione production-grade
│   ├── models.py                     # Definizione architetture con Transfer Learning
│   ├── utils.py                      # Funzioni di supporto (plot, checkpointing, utility)
│   └── app.py                        # Applicazione web Streamlit
│
├── results/
│   └── models/                       # Pesi dei modelli addestrati (.pth) — generati a runtime
│
├── train_phase1.py                   # Entry point training: Fase 1 (Binaria)
├── mlp_train_phase1.py               # Entry point training: Baseline MLP
├── train_phase2.py                   # Entry point training: Fase 2 (Multiclasse)
├── requirements.txt
└── README.md
```

---

## Metodologia

### Data Engineering

La gestione robusta dei dati è il fondamento dell'intera pipeline. Le scelte chiave includono:

**Classe Dataset Personalizzata (`AlzheimerDataset`)**
Una sottoclasse di `torch.utils.data.Dataset` gestisce il re-mapping dinamico delle etichette *al volo*, adattandosi tra task binario e multiclasse senza richiedere copie ridondanti del dataset in memoria.

**Split Riproducibile dei Dati**
`random_split` con seed fisso garantisce esperimenti completamente riproducibili. La suddivisione adottata è **70% Train / 15% Validation / 15% Test**. Il test set è segregato rigorosamente e non viene mai consultato durante l'addestramento o la selezione degli iperparametri.

**Strategia di Data Augmentation**
Si applicano il preprocessing standard (resize a `224×224`, `ToTensor`) e la normalizzazione con media e deviazione standard di ImageNet su tutti e 3 i canali. Questo passaggio è un requisito matematico obbligatorio per non sfalsare l'attivazione dei pesi durante il Transfer Learning, anche nel caso di immagini MRI in scala di grigi caricate come tensori a 3 canali.

> **Nota clinica:** Le trasformazioni anatomicamente implausibili (es. flip verticale) sono deliberatamente escluse. Un cervello umano non si presenta mai capovolto in una scansione MRI clinica.

---

### Architetture Neurali ("Model Zoo")

Tutte le architetture sono definite in `src/models.py`. I pesi pre-addestrati su ImageNet vengono caricati e successivamente sottoposti a **fine-tuning** sul dataset MRI.

#### MLP — Baseline
Una rete fully-connected che opera su vettori 1D di pixel appiattiti (150.528 valori). La struttura spaziale dell'immagine viene intenzionalmente distrutta per fungere da *lower bound* matematico, dimostrando l'inadeguatezza dei layer lineari sui dati visivi rispetto alle convoluzioni.

#### MobileNetV2
Una CNN leggera ed efficiente basata su *Depthwise Separable Convolutions*. Scelta come candidato realistico per il deploy in ambienti con risorse computazionali limitate (es. tablet ospedalieri o dispositivi edge).

#### ResNet-18
Una CNN solida con connessioni residue (*skip connections*) che prevengono la degradazione del gradiente durante l'addestramento profondo. Affidabile e consistente in entrambi i task per l'estrazione di feature mediche complesse.

#### ViT-B/16 — Vision Transformer
Modello allo stato dell'arte che elabora le immagini come sequenze di patch `16×16` tramite meccanismi di Self-Attention. Incluso per confrontare il paradigma CNN con quello dei Transformer e per valutare i limiti hardware in ambienti accademici standard.

---

### Training Engine

Il cuore del ciclo di addestramento (`src/engine.py`) implementa funzionalità tipiche degli ambienti di produzione, andando ben oltre un classico ciclo `for` accademico:

| Funzionalità | Descrizione |
|---|---|
| **Model Checkpointing** | I pesi migliori (`.pth`) vengono salvati esclusivamente al miglioramento della Validation Loss |
| **Early Stopping** | L'addestramento si interrompe automaticamente dopo 5 epoche consecutive senza miglioramenti (`patience=5`) |
| **LR Scheduling** | `ReduceLROnPlateau` riduce il learning rate di un fattore 0.1 in caso di plateau |
| **Gestione Memoria** | Gestione proattiva della VRAM per prevenire errori Out-Of-Memory durante il training del ViT |

---

## Risultati

> Le prestazioni sono valutate attraverso una combinazione di metriche appropriate per contesti di diagnostica clinica: **Precision, Recall, F1-Score (Macro), Matrici di Confusione** e **Curve ROC/AUC**. L'accuratezza globale da sola è insufficiente per descrivere l'efficacia di un modello in campo medico.

### Fase 1 — Classificazione Binaria

| Modello | Accuratezza | Recall (Affetto) | Note |
|---|---|---|---|
| **ResNet-18** | **~98%** | **~99%** | Qualità clinica; minimizza i falsi negativi |
| MLP (Baseline) | < 80% | — | Alto tasso di falsi positivi; inadeguato al deploy |

Il Recall quasi perfetto per la classe "Affetto" è la metrica clinica critica: mancare un vero positivo (non rilevare un cervello malato) ha un costo diagnostico enormemente superiore rispetto a un falso allarme.

### Fase 2 — Classificazione Multiclasse

| Modello | F1-Score (Macro) | Costo Computazionale | Note |
|---|---|---|---|
| **MobileNetV2** | ✅ Migliore | Molto basso | Vincitore del benchmark interno |
| ResNet-18 | Alto | Basso | Prestazioni consistentemente solide |
| ViT-B/16 | Competitivo | Molto alto | Proibitivo su GPU standard (T4 Colab) |
| MLP (Baseline) | Basso | Trascurabile | Lower bound di riferimento |

**Risultato chiave:** La sfida principale nella Fase 2 è il confine morfologico tra le classi **Non Demented** e **Very Mild Demented**, dove le differenze strutturali nel tessuto cerebrale sono talmente sottili da indurre in errore tutti i modelli testati. Questo fenomeno è chiaramente visibile nelle voci fuori dalla diagonale principale delle matrici di confusione 4×4.

**ViT vs. MobileNetV2:** Il Vision Transformer ha richiesto un batch size drasticamente ridotto per adattarsi alla GPU T4 di Colab, allungando considerevolmente i tempi di addestramento senza superare in modo consistente la molto più leggera MobileNetV2. Questo conferma che la complessità architetturale non si traduce necessariamente in guadagni di performance su dataset medici di dimensioni moderate.

---

## Applicazione Web

Il ciclo di vita del progetto si conclude con il deploy in produzione tramite **Streamlit** (`src/app.py`). L'interfaccia funziona come un **Sistema di Supporto alle Decisioni Cliniche (CDSS)**:

1. **Selezione del Modello** — Scegli l'architettura desiderata dalla barra laterale per confrontare il comportamento predittivo tra i diversi modelli addestrati.
2. **Caricamento della MRI** — Carica direttamente una scansione cerebrale dal proprio dispositivo.
3. **Inferenza in Tempo Reale** — L'immagine viene pre-processata in un tensore e passata in avanti attraverso la rete selezionata.
4. **Output Probabilistico** — I risultati vengono visualizzati con barre di probabilità Softmax per tutte le classi, permettendo di valutare la confidenza del modello e analizzare i casi limite.

---

## Installazione

### Prerequisiti

| Requisito | Versione / Specifiche |
|---|---|
| Python | ≥ 3.10 |
| GPU con supporto CUDA | Raccomandata (CPU supportata) |
| VRAM | ≥ 4 GB (≥ 8 GB per ViT-B/16) |

### Configurazione dell'Ambiente

**1. Clona la repository**
```bash
git clone https://github.com/MaccarroneAlessia/unict-dlcmm-Maccarrone-Brancaforte-Cassia
cd alzheimer-mri-classification
```

**2. Crea un ambiente virtuale (consigliato)**
```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```

**3. Installa le dipendenze**
```bash
pip install -r requirements.txt
```

**4. Scarica il dataset**
Scarica l'[Augmented Alzheimer MRI Dataset](https://www.kaggle.com/) da Kaggle ed estrailo in:
```
data/raw/
```

---

## Utilizzo

> ⚠️ La cartella `results/models/` è inizialmente vuota. L'applicazione web **richiede i pesi dei modelli addestrati** (file `.pth`) per funzionare. Eseguire gli script di training prima di avviare l'app.

### Addestramento

**Fase 1 — Classificazione Binaria**
```bash
python mlp_train_phase1.py    # Baseline MLP
python train_phase1.py        # ResNet-18
```

**Fase 2 — Classificazione Multiclasse**
```bash
python train_phase2.py        # Tutti i modelli multiclasse
```

I pesi addestrati vengono salvati automaticamente in `results/models/` ad ogni miglioramento della Validation Loss.

### Avvio dell'Applicazione Web
```bash
streamlit run src/app.py
```

### Analisi Esplorativa
Apri i notebook in ordine per una guida completa all'analisi dei dati e dei risultati:
```bash
jupyter notebook notebooks/
```

---

## Limitazioni

### Data Leakage Strutturale

Esiste una limitazione strutturale nota derivante dalla natura del dataset sorgente. L'**Augmented Alzheimer MRI Dataset** è stato pre-augmentato *offline* dal suo autore originale prima della pubblicazione. Di conseguenza, applicare `random_split` a livello di immagine (anziché a livello di paziente) implica che varianti augmentate della *stessa scansione dello stesso paziente* (es. copie ruotate o con contrasto modificato) possano comparire sia nel training set che nel test set.

**Impatto:** Questo gonfia artificialmente le metriche di accuratezza riportate. I valori di performance non riflettono pienamente la reale capacità di generalizzazione su pazienti mai visti.

**Approccio corretto in contesto reale:** In un setting clinico reale, tutti i dati devono essere suddivisi a livello di **paziente** (*Patient-Level Split*) prima di applicare qualsiasi tecnica di augmentation. Questo garantisce che il modello non venga mai valutato su varianti di scansioni già osservate durante il training.

**Scopo di questo progetto:** Ai fini di questo esercizio accademico, le immagini sono state trattate come campioni indipendenti per mantenere il focus sull'ingegnerizzazione della pipeline MLOps, piuttosto che sulla curazione del dataset. Questa limitazione è pienamente riconosciuta e documentata.

---

## Relazione Tecnica

Una relazione tecnica completa con setup sperimentale dettagliato, risultati quantitativi estesi, discussione approfondita e considerazioni cliniche è disponibile nel file:

📄 `RelazioneDeepLearningCMM - NeuroVisionDL.pdf`

---

## Autori

| Nome | Contributo |
|---|---|
| Massimiliano Cassia | Architetture dei modelli, pipeline di training, benchmarking |
| Alessia Maccarrone | Data engineering, analisi esplorativa, gestione repository |
| Martina Brancaforte | Applicazione web, visualizzazione risultati, documentazione |

**Istituzione:** Università degli Studi di Catania (UniCT)
**Corso:** Deep Learning
**Anno Accademico:** 2024/2025

---

<div align="center">
<sub>Realizzato con PyTorch · Streamlit · ❤️ presso UniCT</sub>
</div>
