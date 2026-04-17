# 🚢 Titanic Survival Exploration — Manual Decision Trees to scikit-learn

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/Udacity-ML_Nanodegree-02B3E4)

## 🚀 Overview

Two complementary approaches to predicting Titanic passenger survival. The first notebook builds a decision tree manually through iterative data exploration — starting from a 61.62% baseline (predict nobody survives) and progressively adding conditions on Sex, Age, Pclass, and Embarked to reach 83.61% accuracy. The second notebook applies scikit-learn's DecisionTreeClassifier and RandomForestClassifier with GridSearchCV hyperparameter tuning, achieving 85.47% test accuracy.

## 📊 Results

| Approach | Accuracy |
|---|---|
| Baseline (predict no survivors) | 61.62% |
| Manual: Sex only | 78.68% |
| Manual: Sex + Age < 10 | 79.24% |
| Manual: Sex + Age + Pclass + Embarked | **83.61%** |
| scikit-learn: DecisionTree (GridSearchCV) | **85.47%** |
| scikit-learn: RandomForest (100 trees) | 82.68% |

## ✨ Key Features

**Notebook 1 — Manual Decision Tree (`titanic_survival_exploration.ipynb`):**
- Iterative rule building: each step adds a feature condition and measures accuracy improvement
- Survival statistics visualization at each split to guide feature selection
- Final hand-crafted model uses Sex, Age, Pclass, and Embarked port as decision criteria

**Notebook 2 — scikit-learn (`titanic_survival_exploration_using_scikit_learn.ipynb`):**
- DecisionTreeClassifier with GridSearchCV tuning over `max_depth` and `min_samples_leaf`
- RandomForestClassifier (100 estimators) with feature importance visualization
- Feature importance ranking via Seaborn bar plot showing which features drive predictions

## 🧠 Technical Highlights

- **Progressive Model Building** — The manual notebook demonstrates how each feature incrementally improves accuracy: Sex alone jumps from 61.62% → 78.68% (+17%), then Age adds +0.56%, then Pclass/Embarked adds +4.37%. This shows which features carry the most predictive signal
- **GridSearchCV Optimization** — DecisionTree tuned over `max_depth` [2–10] and `min_samples_leaf` [2–10] using F1-score as the scoring metric. Best parameters: `max_depth=6`, `min_samples_leaf=6`
- **Overfitting Analysis** — Training accuracy (87.08%) vs. test accuracy (85.47%) shows minimal overfitting after hyperparameter tuning, compared to the untuned model
- **Feature Importance** — Random Forest's `.feature_importances_` reveals the relative contribution of each feature, visualized as a ranked bar chart

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML | scikit-learn (DecisionTreeClassifier, RandomForestClassifier, GridSearchCV) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, custom visuals.py |
| Environment | Jupyter Notebook |

## 🏗 Manual Decision Tree Logic

```
Passenger
    │
    ├── Female?
    │   ├── Yes → Survived (majority)
    │   └── No (Male)
    │       ├── Age < 10?
    │       │   ├── Yes → Survived
    │       │   └── No
    │       │       ├── Pclass == 1 or 2?
    │       │       │   └── Check Embarked port
    │       │       └── Pclass == 3 → Did not survive
    │       └── ...
    │
    Accuracy: 83.61%
```

## 📊 Dataset

The Titanic dataset contains 891 passenger records with 10 features.

| Feature | Description |
|---|---|
| Pclass | Passenger class (1st, 2nd, 3rd) |
| Sex | Male / Female |
| Age | Age in years |
| SibSp | # siblings/spouses aboard |
| Parch | # parents/children aboard |
| Fare | Ticket price |
| Embarked | Port (C=Cherbourg, Q=Queenstown, S=Southampton) |

**Target:** Survival (0 = No, 1 = Yes). Class distribution: ~62% did not survive.

## ⚡ Getting Started

```bash
git clone https://github.com/jashjain21/Titanic_survival_exploration.git
cd Titanic_survival_exploration

pip install numpy pandas matplotlib scikit-learn seaborn

# Manual decision tree approach
jupyter notebook titanic_survival_exploration.ipynb

# scikit-learn approach
jupyter notebook titanic_survival_exploration_using_scikit_learn.ipynb
```

## 🔍 What This Project Demonstrates

- **Intuition Before Algorithms** — Building a classifier manually by exploring data statistics before applying ML libraries, showing understanding of why features matter
- **Feature Selection Impact** — Quantifying exactly how much each feature contributes to accuracy (Sex: +17%, Age: +0.56%, Pclass+Embarked: +4.37%)
- **Hyperparameter Tuning** — GridSearchCV with custom F1 scorer to find optimal tree depth and leaf size
- **Model Comparison** — Decision Tree vs. Random Forest on the same data, with feature importance analysis

## 🚧 Limitations / Future Improvements

- **No Feature Engineering** — Title extraction from names (Mr/Mrs/Miss), family size (SibSp + Parch), or fare binning could improve accuracy
- **Missing Value Handling** — Age has missing values; imputation strategies (median, model-based) aren't explored in depth
- **No Cross-Validation in Manual Notebook** — The manual approach evaluates on the full dataset (no train/test split), so the 83.61% may overestimate generalization
- **Limited Model Variety** — Only tree-based models are tried; Logistic Regression, SVM, or ensemble stacking could be compared
