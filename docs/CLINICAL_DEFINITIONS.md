# Clinical Outcome Definitions

## Purpose

This document defines the cardiometabolic outcomes used in the NHANES uncertainty-aware screening project.

The goal is to identify latent or undiagnosed disease using objective examination and laboratory evidence, while separating previously diagnosed disease from undiagnosed disease.

---

## 1. Undiagnosed Diabetes

### Known diabetes

A participant is considered to have known diabetes if they report that a doctor or health professional told them they had diabetes.

NHANES variable:
- `DIQ010`

### Biomarker-defined diabetes

A participant is considered to have biomarker evidence of diabetes if either:

- HbA1c ≥ 6.5%, or
- Fasting plasma glucose ≥ 126 mg/dL

Variables:
- `LBXGH`
- `LBXGLU`

### Undiagnosed diabetes

Undiagnosed diabetes is defined as:

Known diabetes = No  
AND  
Biomarker-defined diabetes = Yes

---

## 2. Undiagnosed Hypertension

### Known hypertension

A participant is considered to have known hypertension if they report that a doctor or health professional told them they had high blood pressure.

NHANES variable:
- `BPQ020`

### Measured hypertension

Measured hypertension is defined as:

- Mean systolic blood pressure ≥ 130 mmHg, or
- Mean diastolic blood pressure ≥ 80 mmHg

Variables:
- `BPXSY1`, `BPXSY2`, `BPXSY3`
- `BPXDI1`, `BPXDI2`, `BPXDI3`

### Undiagnosed hypertension

Undiagnosed hypertension is defined as:

Known hypertension = No  
AND  
Measured hypertension = Yes

---

## 3. Undiagnosed Dyslipidemia

### Known high cholesterol

A participant is considered to have known high cholesterol if they report being told by a doctor or health professional that their cholesterol was high.

NHANES variable:
- `BPQ080`

### Measured dyslipidemia

Measured dyslipidemia is defined as any of:

- Total cholesterol ≥ 240 mg/dL
- HDL cholesterol < 40 mg/dL
- Triglycerides ≥ 200 mg/dL

Variables:
- `LBXTC`
- `LBDHDD`
- `LBXTR`

### Undiagnosed dyslipidemia

Undiagnosed dyslipidemia is defined as:

Known high cholesterol = No  
AND  
Measured dyslipidemia = Yes

---

## 4. Possible CKD Risk

CKD risk is defined using albuminuria as a proxy:

- Albumin-creatinine ratio ≥ 30 mg/g

Variables:
- `URXUMA`
- `URXUCR`

This is labeled as possible CKD risk rather than confirmed CKD because full CKD staging requires eGFR and/or persistent abnormal kidney markers over time.

---

## 5. Any Latent Cardiometabolic Disease

A participant is considered to have any latent cardiometabolic disease if any of the following are present:

- Undiagnosed diabetes
- Undiagnosed hypertension
- Undiagnosed dyslipidemia
- Possible CKD risk

---

## Important Modeling Rule: Avoid Diagnostic Leakage

A variable used to define an outcome should not be used as a predictor for that same outcome in strict screening models.

Examples:

- HbA1c and fasting glucose should not be used to predict undiagnosed diabetes.
- Blood pressure measurements should not be used to predict undiagnosed hypertension in the strict survey-only model.
- Cholesterol and triglycerides should not be used to predict dyslipidemia in strict screening models.
- Albumin-creatinine ratio should not be used to predict CKD risk.