# Non-Invasive Detection of Diabetes Using a Multimodal Approach

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Détection non-invasive du diabète de type 1** par fusion tardive de signaux **ECG** et **respiratoires**, exploitant les dysfonctionnements du système nerveux autonome (SNA).

**Auteurs :** Wael MELKI & Youssef CHAARI
**Encadrante :** Mme. Sofia BEN JEBARA
**Institution :** École Supérieure des Communications de Tunis (Sup'Com), Université de Carthage
**Année académique :** 2025-2026

---

## Problématique

Le diabète est une maladie métabolique chronique touchant des millions de personnes. Les méthodes de diagnostic traditionnelles (glycémie à jeun, OGTT, HbA1c) sont **invasives**, douloureuses et inadaptées au suivi continu.

Ce projet propose un cadre **non-invasif** de détection du diabète de type 1 à partir de signaux physiologiques reflétant l'activité du système nerveux autonome (SNA).

---

## Jeu de Données : DINAMO

Le dataset **DINAMO** ([ECG and Glucose Dataset](https://www.kaggle.com/)) contient des enregistrements physiologiques de **29 sujets** :

- **20** individus sains
- **9** individus avec diabète de type 1

**Modalités utilisées :** ECG (250 Hz) et signal respiratoire (18 Hz).

---

## Méthodologie

```
┌─────────────────────────────────────────────────────┐
│               Fusion Tardive (Late Fusion)           │
│   P_fusion = α · P_ECG + (1 - α) · P_RESP           │
│   α optimal = 0.10                                    │
└─────────────────────────────────────────────────────┘
         ▲                    ▲
         │                    │
┌────────┴────────┐  ┌────────┴────────┐
│   Pipeline ECG  │  │ Pipeline Respiratoire│
├─────────────────┤  ├─────────────────┤
│ SVM (noyau RBF) │  │ Random Forest   │
│ C=100, balanced │  │ 1000 arbres     │
├─────────────────┤  ├─────────────────┤
│ 4 features HRV  │  │ 18 features     │
│ (mean_rr,       │  │ (temporelles,   │
│  std_rr,        │  │  amplitude,     │
│  rmssd, mean_hr)│  │  spectrales)    │
└─────────────────┘  └─────────────────┘
```

### 1. Pipeline ECG

**Prétraitement :** Filtre Butterworth passe-bande 0.5–50 Hz (ordre 1).
**Détection des pics R :** Algorithme de Pan-Tompkins simplifié (dérivation, quadrature, moyenne glissante).
**Extraction de caractéristiques HRV :**

| Feature | Description |
|---------|-------------|
| `mean_rr` | Intervalle RR moyen |
| `std_rr` | SDNN — Écart-type des intervalles RR |
| `rmssd` | RMSSD — Racine carrée de la moyenne des différences au carré |
| `mean_hr` | Fréquence cardiaque moyenne |

**Classification :** SVM à noyau RBF, `C=100`, `class_weight='balanced'`, optimisation AUC par `GroupKFold` (sujet-aware).

### 2. Pipeline Respiratoire

**Prétraitement :** Filtre Butterworth passe-bande 0.05–0.7 Hz (ordre 4).
**Détection des cycles :** Identification des pics inspiratoires et des creux expiratoires.
**Extraction de caractéristiques (18) :**

| Catégorie | Features |
|-----------|----------|
| **Temporelles** | `BR_mean`, `BR_std`, `RMSSD`, `SDNN`, `CV`, `Ti_mean`, `Ti_std`, `Te_mean`, `Te_std`, `IE_ratio` |
| **Amplitude** | `Amp_mean`, `Amp_std` |
| **Spectrales** | `Dominant_Freq`, `Power_Low_Freq` (0.05–0.2 Hz), `Power_Med_Freq` (0.2–0.5 Hz), `Power_High_Freq` (0.5–0.7 Hz), `Spectral_Entropy`, `Spectral_Centroid` |

**Classification :** Random Forest (1000 arbres, `balanced_subsample`, LOOCV + optimisation de seuil par balanced accuracy).

### 3. Fusion Tardive

$$P_{fused} = \alpha \cdot P_{ECG} + (1 - \alpha) \cdot P_{RESP}$$

- **α optimal = 0.10** (recherche par grille sur les probabilités OOF, maximisant le rappel diabétique)
- Split sujet-aware strict (aucune fuite entre train et test)

---

## Résultats

| Modèle | Accuracy | Précision (Diab.) | Rappel (Diab.) | Spécificité | AUC |
|--------|:--------:|:-----------------:|:--------------:|:-----------:|:---:|
| **Respiratoire seul** | 76.9% | 70.0% | 70.0% | 81.2% | — |
| **ECG seul** | 73.0% | 62.0% | 80.0% | 69.0% | 0.6875 |
| **Fusion multimodale** | **80.8%** | **74.0%** | **70.0%** | **87.5%** | **0.7437** |

La fusion tardive améliore significativement les performances globales, en particulier la **spécificité** (87.5%) et l'**AUC** (0.7437).

---

## Structure du Répertoire

```
├── tutoré breathing.ipynb       # Pipeline respiratoire complet (RF, LOOCV, optimisation de seuil)
├── tutoré ECG.ipynb             # Pipeline ECG complet (SVM, GridSearch, Youden, ablation)
├── feature extraction.ipynb     # Appariement ECG-respiratoire et extraction des features
├── fusion.ipynb                 # Fusion tardive et optimisation de α
├── features_train_df.csv        # Features respiratoires (train)
├── features_test_df.csv         # Features respiratoires (test)
├── df_ecg_train.csv             # Features ECG (train)
├── df_ecg_test.csv              # Features ECG (test)
├── README.md                    # Ce fichier
└── LICENSE                      # MIT License
```

### Notebooks

| Notebook | Description |
|----------|-------------|
| `feature extraction.ipynb` | Appariement des sessions ECG-respiratoire par sujet, split train/test sujet-aware, extraction des 4 features ECG (HRV) et 18 features respiratoires |
| `tutoré breathing.ipynb` | Normalisation MinMax, pondération par sujet, Random Forest (1000 arbres), optimisation de seuil par LOOCV, évaluation test |
| `tutoré ECG.ipynb` | Prétraitement, détection R-peaks (Pan-Tompkins), extraction HRV sur GPU (CuPy), SVM avec GridSearch AUC, seuil de Youden, analyse d'importance et ablation |
| `fusion.ipynb` | Probabilités OOF pour les deux modalités, recherche de α optimal (0.10) maximisant le rappel diabétique, évaluation fusionnée |

---

## Installation & Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/WaelMelkiMelki/Non-invasive-detection-of-diabetes-using-a-multimodal-approach.git
cd Non-invasive-detection-of-diabetes-using-a-multimodal-approach

# Installer les dépendances
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Optionnel : accélération GPU pour l'ECG
pip install cupy joblib tqdm
```

Exécutez les notebooks dans l'ordre : `feature extraction.ipynb` → `tutoré breathing.ipynb` / `tutoré ECG.ipynb` → `fusion.ipynb`.

---

## Références

1. *Guidelines and Recommendations for Laboratory Analysis in the Diagnosis and Management of Diabetes Mellitus.*
2. *Integrated cardiovascular/respiratory control in type 1 diabetes evidences functional imbalance.*
3. *DINAMO (ECG and Glucose Dataset)* — [Kaggle](https://www.kaggle.com/)

---

## Licence

Ce projet est sous licence MIT — voir le fichier [LICENSE](LICENSE).
