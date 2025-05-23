# 🩺 Diabetes Prediction using Machine Learning

This project explores and evaluates multiple supervised machine learning algorithms to predict diabetes using the **Diabetes Health Indicators Dataset** provided by the CDC (BRFSS 2015). The project covers the full ML pipeline, including data cleaning, feature engineering, model training, evaluation, and performance comparison.

---

## 📊 Project Highlights

- 📁 Dataset: [Diabetes Health Indicators Dataset (Kaggle)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- 🔢 Samples: 253,680 records | 22 features
- 🎯 Task: Binary classification – Predict if a person has diabetes
- ✅ Focus: High recall and F1-score to catch diabetic cases (handling class imbalance)

---

## 🧠 What This Project Covers

### ✅ Part A–D: Full ML Pipeline
- Data integrity check and missing value handling
- Exploratory Data Analysis (EDA) with plots and insights
- Feature engineering: interaction terms, scaling, bucketing
- Model training: 9 supervised ML models + 1 baseline
- Imbalance handling using `class_weight='balanced'` and evaluation beyond accuracy

### 📊 Models Evaluated
| Model             | F1 Score | Notes                       |
|------------------|----------|-----------------------------|
| XGBoost (Fast)    | **0.47** | Best performing model       |
| Logistic Regression | 0.44   | Lightweight, interpretable  |
| Random Forest     | 0.44     | Ensemble, stable results    |
| Naive Bayes       | 0.41     | High recall, lower precision|
| Dummy Stratified  | 0.14     | Baseline                    |

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

🔮 Future Work

✅ Add unsupervised clustering to discover patient subgroups

🧪 Apply SMOTE or other resampling techniques

🎛️ Build a Streamlit or Gradio interface for real-time risk scoring

🌐 Deploy as an interactive diabetes risk assessment tool

