# 🫀 Heart Disease Prediction using Machine Learning

This project focuses on predicting the presence of heart disease in patients using various health indicators. I trained multiple machine learning models, evaluated their performance using classification metrics, and applied hyperparameter tuning techniques to select the best-performing model.

---

## 🔍 Project Overview

📈 Cardiovascular diseases are a leading cause of death globally. Predicting heart disease early can enable better medical decisions and preventive care. This project explores classical machine learning approaches on structured clinical data to classify whether a patient is at risk of heart disease.

💡 **Key Idea**: Use multiple classifiers and tune them using `GridSearchCV` and `RandomizedSearchCV` to build a robust, accurate, and interpretable heart disease prediction system.

---

## 📁 Project Structure

```
heart-disease-prediction-ml/
├── heart_disease_prediction.ipynb     # Main notebook with code
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── dataset/
│   └── heart.csv                      # Input dataset
└── best_heart_disease_model.pkl       # Saved best-performing model
```

---

## 📊 Dataset Description

* **Source**: Kaggle / UCI - Heart Disease Dataset
* **Target Variable**: `HeartDisease` (1 = disease present, 0 = no disease)
* **Total Records**: 918
* **Features**:

  * `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`
  * `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`

---

## 🎯 Goals

* Load and clean structured clinical data
* Train multiple machine learning models
* Evaluate performance using classification metrics
* Tune hyperparameters using `GridSearchCV` and `RandomizedSearchCV`
* Select and save the best-performing model

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
```

2. **Create a virtual environment**

```bash
python3 -m venv tfenv
source tfenv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add dataset**
   Place the `heart.csv` file inside the `dataset/` folder.

5. **Run the notebook**

```bash
jupyter notebook heart_disease_prediction.ipynb
```

---

## 🧠 Models Used

| Model                        | Description                  |
| ---------------------------- | ---------------------------- |
| Logistic Regression          | Linear baseline              |
| Support Vector Machine (SVM) | Margin-based classifier      |
| K-Nearest Neighbors (KNN)    | Distance-based model         |
| Decision Tree                | Simple tree-based classifier |
| Random Forest                | Ensemble of decision trees   |

---

## 🧪 Evaluation Metrics

I used the following metrics for model evaluation:

* ✅ Accuracy
* ✅ Precision
* ✅ Recall
* ✅ F1-Score

These were compared across all models, and the best scores were selected.

---

## 🔍 Hyperparameter Tuning

### 🧩 Grid Search (on Random Forest)

* Exhaustively searched:

  * `n_estimators`: \[50, 100, 150]
  * `max_depth`: \[None, 10, 20]
  * `min_samples_split`: \[2, 5]

### 🎲 Randomized Search (on Random Forest)

* Sampled random combinations from:

  * `n_estimators`: 50–200
  * `max_depth`: \[None, 5, 10, 15, 20]
  * `min_samples_split`: 2–10
  * `min_samples_leaf`: 1–4
  * `max_features`: \['sqrt', 'log2', None]

---

## 📊 Results Summary

| Model                              | F1-Score       |
| ---------------------------------- | -------------- |
| Logistic Regression                | 0.87           |
| Decision Tree                      | 0.86           |
| Random Forest                      | 0.89           |
| Random Forest (GridSearchCV)       | 0.90           |
| Random Forest (RandomizedSearchCV) | ⭐️ **0.91** ⭐️ |

➡️ **Best Model**: Random Forest (RandomizedSearchCV)

---

## 🧠 Feature Importance (Sample)

| Rank | Feature        | Importance |
| ---- | -------------- | ---------- |
| 1    | ChestPainType  | ⭐️⭐️⭐️⭐️⭐️ |
| 2    | Oldpeak        | ⭐️⭐️⭐️⭐️   |
| 3    | MaxHR          | ⭐️⭐️⭐️     |
| 4    | ST\_Slope      | ⭐️⭐️⭐️     |
| 5    | ExerciseAngina | ⭐️⭐️⭐️     |

Printed via:

```python
model.feature_importances_
```

---

## 📝 Output & Model Saving

* Final model saved as:

```bash
best_heart_disease_model.pkl
```
---

## 🚀 Future Enhancements

* ✅ Add confusion matrix & ROC curves
* ✅ Deploy using Streamlit for real-time prediction
* 🔧 Try XGBoost, LightGBM, and ensemble voting
* 📈 Include SHAP for model explainability
* 🔍 Add feature engineering on age bins, cholesterol ratios, etc.

---

## 🙌 Acknowledgments

* Kaggle Heart Disease Dataset
* scikit-learn documentation
* Inspired by real-world clinical ML applications

---


