# Non-Invasive Detection of Diabetes Using a Multimodal Approach

**Authors:** Wael MELKI & Youssef CHAARI  
**Supervisor:** Mrs. Sofia BEN JEBARA  
**Institution:** Higher School of Communications of Tunis (Sup'Com), University of Carthage  
**Academic Year:** 2025-2026

## 📌 Project Overview

Diabetes is a chronic metabolic disorder affecting millions worldwide. Traditional diagnosis methods (fasting plasma glucose, OGTT, HbA1c) are invasive, painful, and unsuitable for continuous monitoring.

This project proposes a **non-invasive, multimodal machine learning framework** to detect **Type 1 Diabetes** using physiological signals that reflect Autonomic Nervous System (ANS) activity:
1.  **ECG Signals** (Cardiac activity)
2.  **Breathing Signals** (Respiratory dynamics)

By fusing these two modalities, we aim to capture the subtle autonomic dysfunctions associated with diabetes to create an accessible and comfortable screening tool.

---

## 📊 Dataset

We utilized the **DINAMO Dataset**, which includes physiological recordings from **29 subjects**:
* 20 Healthy individuals
* 9 Individuals with Type 1 Diabetes

The project focuses exclusively on the ECG and Breathing signal modalities provided by this dataset.

---

## 🛠️ Methodology & Architecture

Our solution employs a **Late Fusion** strategy combining two independent processing pipelines.

### 1. Breathing Signal Pipeline
* **Preprocessing:** * Bandpass filtering ($f_{low}=0.05 Hz, f_{high}=0.7 Hz$) to remove baseline drift and high-frequency noise.
    * Breathing Cycle Detection (Inspiratory peaks and Expiratory troughs).
* **Feature Extraction:**
    * **Time-Domain:** Breath-to-Breath Interval (BBI), Respiratory Rate (Mean, Std).
    * **Variability:** RMSSD, SDNN, Coefficient of Variation (CV).
    * **Phase Timing:** Inspiratory/Expiratory times (Ti, Te), I/E Ratio.
    * **Frequency-Domain:** Power Spectral Density (PSD) using Welch’s method, Spectral Entropy, Spectral Centroid.
* **Classification:** Random Forest Classifier with Leave-One-Out Cross-Validation (LOOCV).

### 2. ECG Signal Pipeline
* **Preprocessing:**
    * Butterworth Bandpass filtering ($0.5 - 50 Hz$).
    * **R-Peak Detection:** Implemented using a simplified **Pan-Tompkins algorithm** (differentiation, squaring, moving average).
* **Feature Extraction:**
    * **HRV Analysis:** RR Intervals, RMSSD, Mean Heart Rate, Standard Deviation of RR intervals.
* **Classification:** Support Vector Machine (SVM) with RBF kernel and probability estimates.

### 3. Multimodal Fusion
We implemented a weighted late fusion of the probability scores from both classifiers:

$$P_{fused} = \alpha \cdot P_{ECG} + (1 - \alpha) \cdot P_{RESP}$$

* The optimal weight $\alpha$ was determined to be **0.10** via grid search to maximize diabetic recall.

---

## 📈 Results

The multimodal approach demonstrated superior performance compared to unimodal models, confirming that integrating cardiac and respiratory dynamics enhances detection capability.

| Model | Accuracy | Diabetic Recall | Notes |
| :--- | :---: | :---: | :--- |
| **Breathing Only** | 77.0% | Moderate | Good baseline identification |
| **ECG Only** | 73.0% | 80.0% | High sensitivity to diabetic cases |
| **Multimodal Fusion** | **80.8%** | **Balanced** | Best overall performance |

### Confusion Matrix (Fusion Model)
*(Please insert Figure 5 from the report here)*

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* NumPy
* Pandas
* SciPy (for signal processing filters)
* Scikit-learn (for SVM, Random Forest, and metrics)
* Matplotlib/Seaborn (for visualization)

### Running the Project
1.  Clone the repository:
    ```bash
    git clone [https://github.com/WaelMelkiMelki/Non-invasive-detection-of-diabetes-using-a-multimodal-approach.git](https://github.com/WaelMelkiMelki/Non-invasive-detection-of-diabetes-using-a-multimodal-approach.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main pipeline (example):
    ```bash
    python main.py
    ```

---

## 📚 References
1.  *Guidelines and Recommendations for Laboratory Analysis in the Diagnosis and Management of Diabetes Mellitus.*
2.  *Integrated cardiovascular/respiratory control in type 1 diabetes evidences functional imbalance.*
3.  *ID1NAMO (ECG and Glucose Dataset).*

---

**Note:** This project was realized as part of a tutored engineering project at **Sup'Com**.
