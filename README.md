# Health Risk Prediction

## Overview

Build an end-to-end pipeline to predict a student's health risk from lifestyle and physiological features. Part of Kaggle's Playground Series (Season 6, Episode 7).

## Dataset

Source: https://www.kaggle.com/competitions/playground-series-s6e7/data

- Number of observations: 690088
- Number of features: 13 (1 nominal, 5 ordinal, 7 numeric)
- Target classes: fit (39803), at-risk (592561), unhealthy (57724)
- Total missing values: 449496

## Workflow

EDA -> Data Cleaning + Feature Engineering -> Baseline + Model Selection -> Hyperparameter Tuning -> Evaluation

## Features Created

- Missing value count per row
- Activity (encoded) × exercise duration

## Models Compared

| Model | Balanced Accuracy |
|-------|------------------:|
| LogisticRegression | 0.879 |
| RandomForestClassifier | 0.929 |
| HistGradientBoostingClassifier | 0.937 |
| XGBoost | 0.847 |

Final model: HistGradientBoostingClassifier (highest balanced accuracy, quicker to train than RandomForestClassifier on large dataset)

## Results

Leaderboard score: **0.933**

## Key Predictors

1. Sleep duration
2. Stress level
3. BMI
4. Activity × duration (created)
5. Missing value count per observation (created)

## Summary

- Built a preprocessing pipeline with custom feature engineering
- Created two features following EDA which carried predictive value
- Compared four classification models for imbalanced classes using balanced accuracy 
- Used permutation importance to find key predictors

## Future Improvements

- Feature selection could filter noise, leading to better predictions
- Other evaluation metrics could be used (eg. F1 score) to improve model/hyperparameter selection
