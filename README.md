# 🩺 Diabetes Prediction using Machine Learning

This project explores and evaluates multiple supervised machine learning algorithms to predict diabetes using the **Diabetes Health Indicators Dataset** provided by the CDC (BRFSS 2015). The project covers the full ML pipeline, including data cleaning, feature engineering, model training, evaluation, and performance comparison. This project uses supervised machine learning models to predict diabetes based on CDC's BRFSS 2015 dataset. It includes preprocessing, model evaluation, clustering analysis, and a deployable interactive Gradio app.

---

## 📊 Project Highlights

- 📁 Dataset: [Diabetes Health Indicators Dataset (Kaggle)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- 🔢 Samples: 253,680 records | 22 features
- 🎯 Task: Binary classification – Predict if a person has diabetes
- ✅ Focus: High recall and F1-score to catch diabetic cases (handling class imbalance)

---
## 📁 Files Included

| File                     | Description                                        |
|--------------------------|----------------------------------------------------|
| `diabetes_prediction.ipynb` | Full notebook: EDA, modeling, evaluation        |
| `diabetes_gradio_app.py`    | Gradio web app script for diabetes prediction   |
| `xgb_model.pkl`             | Trained XGBoost model                           |
| `scaler.pkl`                | StandardScaler used for input normalization     |
| `requirements.txt`          | Python dependencies                             |
| `README.md`                 | Project overview                                |

---

## 🧠 What This Project Covers

### ✅ Part A–D: Full ML Pipeline
- Data integrity check and missing value handling
- Exploratory Data Analysis (EDA) with plots and insights
- Feature engineering: interaction terms, scaling, bucketing
- Model training: 9 supervised ML models + 1 baseline
- Imbalance handling using `class_weight='balanced'` and evaluation beyond accuracy

### 📊 Models Evaluated
| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| XGBoost (Fast)      | 0.82     | 0.39      | 0.59   | 0.47     |
| Logistic Regression | 0.73     | 0.31      | 0.78   | 0.44     |
| Random Forest       | 0.72     | 0.30      | 0.79   | 0.44     |
| Naive Bayes         | 0.77     | 0.32      | 0.57   | 0.41     |
| Dummy Stratified    | 0.76     | 0.14      | 0.14   | 0.14     |


### 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- ROC curves
- Heatmaps and bar charts

---

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Libraries**:  
  `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `lightgbm`

---


## 🚀 How to Run Locally

# Clone the repository
git clone https://github.com/HazalAtis/diabetes-ml-prediction.git
cd diabetes-ml-prediction

📎 [Click here to open the notebook in Google Colab](https://colab.research.google.com/drive/15bqKRl_2NsfAMXfhcHykiesaIfSQ0nnL?usp=sharing)


---

## 🚀 Live App
---

🔍 Clustering Analysis (Unsupervised Learning)
K-Means clustering was applied using PCA-reduced health features to explore hidden groupings among patients. This can help identify subpopulations with distinct risk profiles or behavior patterns.

---

🔮 Future Enhancements

✅ Deploy interactive app using Hugging Face Spaces

🔄 Improve class balance using SMOTE or class weights

🧮 Experiment with deep learning models (MLP)

🌐 Add language and accessibility support

