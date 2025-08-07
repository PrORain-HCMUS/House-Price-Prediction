
# 🏡 Prediction Interval Competition II: House Price (Kaggle 2025)

<div align="center">
  <a href="https://www.kaggle.com/competitions/prediction-interval-competition-ii-house-price">
    <img src="https://img.shields.io/badge/Kaggle-Competition-blue" alt="Kaggle Badge">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT Badge">
  </a>
</div>



## 📌 Overview

This repository contains our solution to the Kaggle competition: **Prediction Interval Competition II: House Price**, where the goal is to predict **a 90% prediction interval** for house sale prices — not a single point estimate. The narrower and more accurate the interval, the better.

- 📅 **Start**: Apr 23, 2025
- 📅 **End**: Jul 27, 2025
- 🏆 **Evaluation Metric**: [Winkler Score](https://en.wikipedia.org/wiki/Prediction_interval#Winkler_score)
- 🎯 **Goal**: Narrowest prediction intervals while maintaining 90% coverage

## 📂 Folder Structure

```
.
├── processed/               # Processed data (numpy arrays, encoders, scalers)
├── models/                  # Trained models (LightGBM)
├── datasets/                # Base datasets provided by the author
├── src/                     # Python scripts and notebooks (modularized code)
├── submission/              # Final submission CSVs
├── report.pdf               # Final project report (Vietnamese)
├── Slides.pdf               # Presentation slides
└── README.md                # You're here!
```

## 🧠 Methodology

Our approach consists of two key components:

### 1. LightGBM Quantile Regression

We trained two separate LightGBM models to predict the 5th and 95th percentiles of the house price distribution:

`ŷ_lower = Q_0.05(y|x),   ŷ_upper = Q_0.95(y|x)`


This gives us a raw 90% prediction interval.

### 2. Conformal Prediction Adjustment

We then post-process the raw intervals using **Conformal Prediction**:

`residual_i = max(0, ŷ_i^(l) - y_i) + max(0, y_i - ŷ_i^(u))`

`q_hat = Quantile_0.9(residuals)`

`ŷ_i_adj^(l) = ŷ_i^(l) - q_hat`
`ŷ_i_adj^(u) = ŷ_i^(u) + q_hat`


This ensures the **coverage guarantee** without assuming any distribution.

## 📈 Performance Metrics

| Metric              | Value        |
|---------------------|--------------|
| Coverage            | 0.900        |
| Winkler Score       | 341,850.64   |
| Mean Interval Width | 247,135.10   |
| MAE                 | 63,012.56    |
| RMSE                | 116,814.94   |

## 💡 Key Contributions

- ✅ Robust feature engineering: interactions, ratios, log-transforms, spatial features
- ✅ Effective handling of categorical, temporal, and missing data
- ✅ Modular training pipeline with support for LightGBM and Conformal Prediction
- ✅ Submission-ready architecture and reproducibility

## 🧑‍💻 Team Members

| Name                  | Student ID   | Role                             |
|-----------------------|--------------|----------------------------------|
| Lê Đại Hòa            | 22120108     | Project Leader, Slides, Demo     |
| Lê Châu Hữu Thọ       | 22120350     | EDA, Data Processing             |
| Nguyễn Tường Bách Hỷ  | 22120455     | Conformal Prediction, Analysis, Training   |
| Lê Hoàng Vũ           | 22120461     | Modeling, Training, Report       |


## 📎 Resources

- 📄 [Final Report (Vietnamese)](./report.pdf)
- 🖼️ [Presentation Slides](./slides.pdf)

## 🧪 Local Evaluation (Winkler Score)

```python
def winkler_score(y_true, l, u, alpha=0.1):
    width = u - l
    penalty = np.where(y_true < l, (2/alpha)*(l - y_true),
              np.where(y_true > u, (2/alpha)*(y_true - u), 0))
    return width + penalty
```

## 📜 License

This repository is licensed under the [MIT License](./LICENSE).

## 🤝 Acknowledgements

We would like to thank the course instructors and Kaggle organizers for this wonderful opportunity to explore **uncertainty quantification** in machine learning!
