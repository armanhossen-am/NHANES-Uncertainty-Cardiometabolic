# Study Design

## Title
Uncertainty-aware Multi-outcome Screening for Undiagnosed Cardiometabolic Disease in the U.S. Population Using NHANES

## Core Aim
Develop and evaluate an uncertainty-aware deep learning framework for screening latent or undiagnosed cardiometabolic disease using NHANES.

## Primary Outcomes
1. Undiagnosed diabetes
2. Undiagnosed hypertension
3. Undiagnosed dyslipidemia
4. Possible CKD risk
5. Any latent cardiometabolic disease

## Main Contribution
This study is not only a prediction study. It develops a decision-support framework that separates:
- confident low-risk individuals
- confident high-risk individuals
- uncertain individuals who should be referred for confirmatory testing

## Models
Baseline:
- Logistic Regression
- XGBoost

Deep learning:
- Single-task MLP
- Multi-task MLP with shared encoder and separate disease heads

## Uncertainty Method
- Split conformal prediction
- Disease-specific conformal prediction
- Subgroup-aware conformal analysis

## Evaluation
Prediction:
- AUROC
- AUPRC
- sensitivity
- specificity
- F1-score

Calibration:
- Brier score
- calibration curve
- expected calibration error

Uncertainty:
- empirical coverage
- uncertainty rate
- average prediction set size
- risk-coverage curve
- subgroup uncertainty

## Subgroup Analysis
Evaluate uncertainty by:
- age group
- sex
- race/ethnicity
- income
- education
- BMI category

## Main Research Question
Can conformal prediction make deep learning screening models more reliable for identifying undiagnosed cardiometabolic disease in the U.S. population?

## Key Novelty
A multi-outcome uncertainty-aware screening framework for latent cardiometabolic disease, rather than a diabetes-only prediction model.