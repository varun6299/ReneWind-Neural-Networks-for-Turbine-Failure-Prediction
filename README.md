# 🌪ReneWind — Neural Networks for Turbine Failure Prediction
*A neural network–based predictive maintenance solution to identify turbine failures before they happen.*

---

## Project Overview & Motivation
Wind energy has become a critical part of the renewable energy mix, but turbines are complex machines that are costly to repair and maintain. **Unexpected failures not only cause downtime but can cost thousands of dollars per hour in lost productivity**.  

ReneWind, a leading energy provider, wanted to leverage **sensor data collected from turbines** to anticipate failures before they occurred. By building a machine learning–based system, the company aimed to:  
- **Reduce unplanned downtime** through early warnings.  
- **Optimize maintenance schedules** with predictive insights.  
- **Improve safety and reliability** by catching failures before they escalate.  

This project applies **artificial neural networks (ANNs)** to historical turbine data to predict failures. Given the rarity of failures, the business goal was to **maximize recall** — ensuring that as many true failures as possible were identified, even if it meant accepting some false alarms.

---

## Objective
The primary objective is to develop a neural network–based classifier that predicts whether a turbine will **fail (1)** or **not fail (0)**.  

From a business perspective:  
- **False negatives (missed failures)** are extremely costly, since they lead to downtime and expensive repairs.  
- **False positives (incorrectly flagged failures)** are less damaging, since they only trigger additional inspections.  

👉 Therefore, the model was designed to prioritize **recall** while maintaining a good **F1 Score** to balance precision and recall.  

---

## Dataset
- **Training set:** ~20,000 records  
- **Test set:** ~5,000 records  
- **Features:** 40 predictor variables (ciphered/transformed sensor readings)  
- **Target:** `Failure` — binary indicator of turbine failure (1) or normal operation (0)  
- **Class distribution:** Failures were rare compared to non-failures, creating a strong **class imbalance** problem.  

Because features were anonymized, their physical meaning is unknown, but they retain predictive signal for classification.  

---

## Key Data Observations (EDA Highlights)
- **Distribution:** Most predictor variables followed a roughly normal distribution.  
- **Imbalance:** Failures represented only a small percentage of the dataset, making imbalance handling critical.  
- **Failure vs Non-failure patterns:** Certain predictors showed separation between failing and non-failing turbines in boxplots and histograms, suggesting useful signal.  
- **Outliers:** Present but retained, as rare abnormal readings could correspond to early failure indicators.  

These insights guided preprocessing and modeling choices, especially the emphasis on **recall**.

---

## Data Preprocessing
To prepare the dataset for neural network modeling, the following steps were taken:  

- **Train-validation split:** Data split into **70% training and 30% validation**, using stratified sampling to maintain the imbalance ratio.  
- **Scaling:** Features standardized to stabilize gradient descent in neural networks.  
- **Class imbalance handling:**  
  - Baseline models were trained without resampling.  
  - **Class weights** were later applied during training to give more importance to minority class (failures).  

This ensured models learned to detect failures without overfitting to the majority class.

---

## Modeling Approach
Multiple ANN architectures were tested, varying in depth, width, regularization, and optimizer choice:  

- **SGD-based models:** Provided baseline comparisons (`NN_SGD_2Hidden`, `NN_SGD_DeepWide`).  
- **Adam-based models:** Improved convergence speed and stability (`NN_Adam_4Hidden`).  
- **Dropout regularization:** 30% dropout layers added to reduce overfitting.  
- **Weighted models:** Incorporated class weights to penalize missed failures.  

Representative examples:  
- **NN_SGD_2Hidden_Drop30** – a shallow net with dropout.  
- **NN_Adam_4Hidden_Drop30_Weighted** – deeper network with Adam optimizer, dropout, and class weighting (eventually chosen as best).  
- **NN_SGD_DeepWide** – larger architecture to test capacity improvements.  

Each variant was evaluated on recall, precision, and F1 Score to balance predictive performance and business requirements.  

---

## Training & Hyperparameters
- **Optimizers:**  
  - *SGD* (Stochastic Gradient Descent) for baseline interpretability.  
  - *Adam* for adaptive learning rates and faster convergence.  
- **Epochs:** Typically 50–100, with performance monitored to avoid overfitting.  
- **Batch size:** 32–64.  
- **Regularization:** 30% dropout layers reduced overfitting risk.  
- **Class weighting:** Applied to give more importance to minority failure class.  
- **Metrics tracked:** Accuracy, Precision, Recall, and F1 Score (with recall emphasized).  

Training logs confirmed that Adam-based models with class weighting achieved higher recall and F1 compared to their unweighted counterparts.

---

## Evaluation Metrics
- **Accuracy** was not prioritized due to imbalance; it would misleadingly favor majority predictions.  
- **Recall** was the most critical metric: missing a failure (false negative) is costly.  
- **Precision** ensured flagged failures were credible, preventing too many false alarms.  
- **F1 Score** balanced recall and precision, serving as the primary selection criterion.  

👉 **Final model selection was based on F1 Score, with strong preference given to recall.**

---

## Model Comparison Results
Final validation performance across architectures:

| Model                              | Train Recall | Train F1 | Val Recall | Val F1 |
|-----------------------------------|--------------|----------|------------|--------|
| NN_SGD_2Hidden_Drop30             | 0.7093       | 0.7168   | 0.6829     | 0.6872 |
| **NN_Adam_4Hidden_Drop30_Weighted** | 0.7687       | 0.7769   | **0.7291** | **0.7350** |
| NN_SGD_DeepWide                   | 0.7365       | 0.7440   | 0.7012     | 0.7090 |
| NN_Adam_2Hidden                   | 0.7204       | 0.7279   | 0.6937     | 0.6998 |
| NN_SGD_4Hidden                    | 0.7321       | 0.7405   | 0.6985     | 0.7059 |

---

## Final Model Selection
**Chosen model:** `NN_Adam_4Hidden_Drop30_Weighted`  
- Delivered the **highest validation recall (0.7291)** and **F1 Score (0.7350)**.  
- Balanced recall and precision effectively, avoiding both missed failures and excessive false alarms.  
- Outperformed deeper or wider architectures without class weighting.  

This model represents the best trade-off between business needs (don’t miss failures) and operational efficiency (minimize unnecessary checks).

---

## Confusion Matrix & Classification Report
For the final model (`NN_Adam_4Hidden_Drop30_Weighted`):  
- **True Positives (TP):** Majority of actual failures correctly identified.  
- **False Negatives (FN):** Reduced significantly compared to baseline models.  
- **False Positives (FP):** Slightly higher, but acceptable in context of preventive maintenance.  
- **True Negatives (TN):** Most normal operations classified correctly.  

The classification report confirmed strong recall with balanced precision, validating the model for real-world deployment.

---

## Actionable Business Insights & Impact of Model Performance

The selected model, **NN_Adam_4Hidden_Drop30_Weighted**, achieved:  
- **Validation Recall:** 72.9%  
- **Validation F1 Score:** 73.5%  

These numbers translate directly into **business value** for ReneWind:

1. **Reduced Missed Failures (False Negatives):**  
   - Baseline models missed ~35–40% of failures.  
   - Final model reduced this to **~27% missed failures**, meaning nearly **3 out of 4 actual failures** are now caught in advance.  
   - If each missed failure costs ~$50,000 in downtime and repair, this improvement could save **hundreds of thousands of dollars annually**.

2. **Higher Preventive Maintenance Accuracy:**  
   - **True Positives:** The model correctly identifies most actual failures, enabling early preventive maintenance.  
   - **False Positives:** Increased slightly (extra inspections), but these inspections cost far less (~$5,000 per check) than unplanned breakdowns.  
   - Net effect: The trade-off is **financially favorable** — an acceptable number of inspections in exchange for avoiding catastrophic breakdowns.

3. **Operational Efficiency Gains:**  
   - With recall close to 73%, ReneWind can plan **maintenance crews and spare parts** more effectively.  
   - Instead of reacting after breakdowns, technicians can proactively address issues in 3 out of 4 failing turbines.  
   - This improves **turbine uptime**, reduces revenue loss, and builds **grid reliability**.

4. **Strategic Data Recommendations:**  
   - Expanding the dataset with more failure cases will likely push recall above **80%**, further reducing costly downtime.  
   - Combining this model with real-time streaming data (SCADA) could enable **predictive alerts** several days in advance.

---

## Impact Summary
- **73% of turbine failures** can now be predicted before they occur.  
- **Annual downtime costs** reduced significantly by catching more failures in advance.  
- **Return on investment:** Every additional percentage point in recall translates to **fewer breakdowns and higher uptime**, directly boosting profitability.  
- **Business alignment:** The model balances the cost of extra inspections (false positives) with massive savings from avoided failures (false negatives).




---
