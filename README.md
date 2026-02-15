
# Gaussian Bayes Classifier from Scratch (Full Covariance)

## 📌 Project Overview

This project implements a **Gaussian Bayes Classifier from scratch** using a full covariance matrix for each class.
The goal is to understand the mathematical foundation of generative classifiers instead of directly using library implementations.

The model is evaluated on the **Date Fruit Classification Dataset**, which contains multiple numerical features describing different types of dates.

---

## 🧠 Motivation

Most students directly use `sklearn.GaussianNB()` without understanding:

* How priors are computed
* How class-wise means are estimated
* How covariance matrices are formed
* How multivariate Gaussian likelihood works
* How posterior probabilities determine classification

This project builds everything manually to deeply understand:

[
P(C_k | x) \propto P(x | C_k) P(C_k)
]

---

## 📊 Dataset

* Multi-class classification problem
* 7 date fruit varieties:

  * BERHI
  * DEGLET
  * DOKOL
  * IRAQI
  * ROTANA
  * SAFAVI
  * SOGAY
* Only numerical features were used
* Stratified train/validation/test split

---

## ⚙️ Implementation Details

For each class ( C_k ):

1. Compute **prior probability**:
   [
   P(C_k) = \frac{\text{Number of samples in } C_k}{\text{Total samples}}
   ]

2. Compute **mean vector**:
   [
   \mu_k = \text{Mean of feature vectors in } C_k
   ]

3. Compute **covariance matrix**:
   [
   \Sigma_k = \text{Covariance of features in } C_k
   ]

4. Model likelihood using **Multivariate Gaussian Distribution**

5. Compute posterior:
   [
   P(C_k | x) = P(x | C_k) P(C_k)
   ]

6. Predict class with maximum posterior probability

---

## 📈 Model Performance

### ✅ Test Accuracy: **74%**

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.74  |
| Macro F1 Score    | 0.64  |
| Weighted F1 Score | 0.73  |

### Strongly Classified Classes

* DOKOL
* SAFAVI
* ROTANA

### Challenging Class

* BERHI (small sample size and feature overlap)

---


### Key Difference:

| Model               | Covariance Assumption | Feature Independence |
| ------------------- | --------------------- | -------------------- |
| This Implementation | Full Covariance       | No                   |
| GaussianNB          | Diagonal Covariance   | Yes                  |

This implementation is closer to **Quadratic Discriminant Analysis (QDA)** than Naive Bayes.

---

## 📂 Project Structure

```
Gaussian-Bayes-Date-Fruit/
│
├── gaussian_bayes_from_scratch.py
├── Date_Fruit_Datasets.csv
├── requirements.txt
└── README.md
```

---

## 🛠 Requirements

Install dependencies:


pip install numpy pandas matplotlib seaborn scipy scikit-learn


## ▶️ How to Run


python gaussian_bayes_from_scratch.py

The script:

* Trains the custom Gaussian Bayes model
* Evaluates on validation and test sets
* Displays classification report
* Plots confusion matrix heatmap
* Compares with sklearn GaussianNB

---

## 🚀 Key Learnings

* Understanding generative classifiers
* Multivariate Gaussian distribution
* Covariance estimation
* Posterior probability computation
* Difference between full covariance and naive assumption
* Multi-class classification handling

---

## 👨‍💻 Author

Vardhan
B.Tech CSE Student
