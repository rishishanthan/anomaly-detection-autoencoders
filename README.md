# Anomaly Detection with Autoencoders (Dense, LSTM, Conv1D)

Three autoencoder variants for detecting anomalies using **reconstruction error**:
- **Dense AE** for tabular signals
- **LSTM AE** for sequential data
- **Conv1D AE** for local temporal patterns

The notebook provides a clean, comparative pipeline with reproducible metrics and visual diagnostics.

---

## 🔍 Approach
1) **Data prep** (windowing/standardization if time series)
2) **Train AEs** on normal data (or normal-heavy data)
3) **Compute reconstruction error** on val/test
4) **Choose threshold** (e.g., val quantile, Youden’s J, or F1-max)
5) **Evaluate**: ROC-AUC, PR-AUC, Accuracy, F1; visualize error distributions
6) **Explain errors**: overlay recon vs. original; error heatmaps

---

## 🧠 Models
- **Dense AE**: MLP encoder/decoder with bottleneck (e.g., 128→32→128)
- **LSTM AE**: Encoder LSTM → bottleneck → Decoder LSTM
- **Conv1D AE**: Temporal conv blocks + upsampling/transposed conv

**Loss:** MSE  
**Optimizer:** Adam (lr=1e-3)  
**Regularization:** Dropout/weight decay as needed  
**Early stopping** on val reconstruction loss

---

## 📌 Insights
- Picking threshold on validation avoids test leakage
- LSTM AE shines with temporal drift; Conv1D with local bursts
- Robust standardization and window size are key hyperparameters

---

## 📊 Results
All the results from my run which of test, train, validation reults including data analysis and corellations are in notebook.

---

## 📁 Dataset
The dataset I used in the folder called 'NAB'. There are many other datasets in the folder, which can be tried out for improvement and practice purposes.
