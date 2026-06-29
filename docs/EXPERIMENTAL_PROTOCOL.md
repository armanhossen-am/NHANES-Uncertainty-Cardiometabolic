# Experimental Protocol

## Study Aim
Develop an uncertainty-aware screening framework for undiagnosed cardiometabolic disease using NHANES 2011–2014.

## Core Outcomes
1. Undiagnosed diabetes
2. Undiagnosed hypertension
3. Undiagnosed dyslipidemia
4. Possible CKD risk
5. Any latent cardiometabolic disease

## Main Hypothesis
As more clinical information becomes available, model performance improves and uncertainty decreases.

## Feature Sets

### Feature Set A: Survey-only screening
Used for low-cost community screening.

Variables:
- age
- sex
- race/ethnicity
- education
- income poverty ratio
- smoking
- alcohol
- physical activity
- health insurance
- medical history

### Feature Set B: Survey + physical examination
Used for primary-care screening.

Add:
- BMI
- waist circumference
- systolic blood pressure
- diastolic blood pressure

### Feature Set C: Survey + examination + routine labs
Used for enhanced clinical screening.

Add:
- total cholesterol
- HDL cholesterol
- triglycerides
- serum creatinine
- albumin-creatinine ratio

Important:
Do not use the same biomarker that defines the disease outcome as a predictor for that disease.

Examples:
- Do not use HbA1c or fasting glucose to predict undiagnosed diabetes.
- Do not use blood pressure values to predict undiagnosed hypertension in the strict screening model.
- Do not use cholesterol values to predict dyslipidemia in the strict screening model.

## Models
Baseline:
- Logistic Regression
- XGBoost

Deep Learning:
- Single-task MLP
- Multi-task MLP with shared encoder and disease-specific heads

## Uncertainty Method
- Split conformal prediction
- Outcome-specific conformal sets
- Subgroup uncertainty analysis

## Evaluation Metrics
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

## Subgroup Analysis
Evaluate performance and uncertainty by:
- age group
- sex
- race/ethnicity
- income
- education
- BMI category

## Main Paper Contribution
This study is not a diabetes-only prediction paper. It proposes a multi-outcome uncertainty-aware screening framework that identifies:
- confident low-risk individuals
- confident high-risk individuals
- uncertain individuals who should be referred for confirmatory testing