# Project Blueprint

## Title
Confidence-guided Screening of Undiagnosed Cardiometabolic Disease Using Multi-task Deep Learning and Conformal Prediction

## Study Aim
To develop an uncertainty-aware clinical AI framework for screening latent or undiagnosed cardiometabolic disease in the U.S. population using NHANES.

## Rationale
Existing work has already applied neural networks and conformal prediction to diabetes screening in NHANES 2011–2014. Our project extends beyond diabetes by developing a multi-outcome cardiometabolic screening framework with uncertainty-guided clinical referral decisions.

## Primary Outcomes
1. Undiagnosed diabetes
2. Undiagnosed hypertension
3. Undiagnosed dyslipidemia
4. Possible CKD risk
5. Any latent cardiometabolic disease

## Main Innovation
The study is not a disease-only prediction model. It is a clinical decision-support framework that classifies individuals into:

- Confident low-risk
- Confident high-risk
- Uncertain, requiring confirmatory testing

## Dataset
Primary dataset:
- NHANES 2011–2014

Planned external validation:
- NHANES 2015–2016
- NHANES 2017–2018

## Feature Sets

### Feature Set A: Survey-only
Community screening variables.

### Feature Set B: Survey + Examination
Adds BMI, waist circumference, and measured blood pressure.

### Feature Set C: Survey + Examination + Routine Labs
Adds routine laboratory markers while avoiding direct diagnostic leakage.

## Modeling Plan
Baseline models:
- Logistic Regression
- XGBoost

Deep learning:
- Single-task MLP
- Multi-task MLP with shared encoder and disease-specific output heads

## Uncertainty Plan
Use conformal prediction to generate prediction sets and uncertainty-based referral decisions.

Main outputs:
- Empirical coverage
- Prediction set size
- Uncertainty rate
- Risk-coverage curve
- Subgroup uncertainty

## Subgroup Analysis
Evaluate uncertainty and performance across:

- Age group
- Sex
- Race/ethnicity
- Income
- Education
- BMI group

## Planned Figures
1. Study design flowchart
2. Framework diagram
3. Outcome prevalence
4. ROC and PR curves
5. Calibration curves
6. Risk-coverage curves
7. Subgroup uncertainty plots
8. Decision curve analysis

## Planned Tables
1. Cohort characteristics
2. Outcome definitions
3. Feature set definitions
4. Model performance
5. Conformal uncertainty performance
6. Subgroup analysis
7. Sensitivity analysis

## Next Technical Milestones
1. Create clinical outcome definition document
2. Create variable dictionary
3. Redesign preprocessing into Feature Sets A, B, C
4. Train baseline models by feature set
5. Build multi-task neural network
6. Add conformal prediction
7. Generate figures and tables