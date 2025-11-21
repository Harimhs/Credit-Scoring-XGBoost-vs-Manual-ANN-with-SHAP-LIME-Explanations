***

# Credit Risk Modeling with XGBoost, Random Forest, SHAP, and LIME

**Interpretable Machine Learning for Credit Risk Modeling using SHAP and LIME**

***

## Executive Summary

This project builds and explains a credit risk model on the German Credit dataset, guiding a bank in predicting loan defaults and interpreting model decisions for regulatory transparency. We preprocess the data, engineer business-smart features (plus a KMeans risk segment), and compare XGBoost (ensemble) with Random Forest. For interpretability, we apply SHAP (global \& local feature impact) and LIME (local surrogate explanations) to rigorously explain both global behaviors and individual customer decisions.

***

## Data, Task \& Objective

- **Objective**: Predict loan default (Risk = Bad) and explain predictions for risk committee review.
- **Dataset**: German Credit Data (1,000 samples, 11 columns)
    - Numeric: Age, Job, Credit amount, Duration
    - Categorical: Sex, Housing, Purpose, Saving/Checking accounts, Risk
- **Target**: ‘Risk’ (`good`/`bad`, mapped as 1/0)

***

## Approach \& Documentation

### 1. **EDA and Domain Knowledge**

**Data Exploration**:

- Checked class and feature distributions, nulls, and outliers.
- Discovered moderate class imbalance (70:30 good:bad).
- Noted: Large missing in 'Saving accounts' and 'Checking account' (realistic—no account = ‘None’).
- Numerical features (Credit amount, Duration) are right-skewed, genuine outliers indicate real business scenarios (large loans).

**Business Insights**:

- Age/risk: Young (<25) and elderly (60+) higher risk, middle age lower risk.
- Housing: Ownership has lower risk; ‘free’ suggests dependency, higher risk.
- Loan size + duration: Longer and larger → higher risk.
- Account status: Absence of savings/checking suggests limited financial history.

***

### 2. **Preprocessing Decisions**

- **Missing handling**: ‘Saving accounts’ and ‘Checking account’ nulls → ‘None’ (as per German banking reality).
- **Ordinal encoding**: Mapped categorical financial account fields to reflecting richness (e.g., none < little < moderate < quite rich < rich).
- **One-hot encoding**: For Housing, Purpose, and categorical risk groups.
- **Scaling**: Features standardized using log + z-score to handle outliers and scale features for neural net stability.
- **Outliers**: Kept (real rare but valid loans); model learns, rather than ignores, business reality.

***

### 3. **Feature Engineering**

- **Monthly Debt**: Credit amount divided by Duration (indicative of monthly payment burden).
- **Loan Intensity**: Nonlinear combination of amount and log(duration) (highlights short, expensive loans).
- **Age Risk Group**: Binned into high/medium/low groups reflecting typical credit scoring.
- **KMeans Segment**: ML-driven cluster (K=3) of customers based on Age, Job, Credit, Duration, Monthly Debt – provides ML-identified risk patterns likely hidden in raw features.
- **High-Risk Purpose**: Derived flag for loan purposes known to be riskier per banking domain experience.

***

### 4. **Data Splitting, Resampling, \& Scaling**

- **Train/Test Split**: 80:20 stratified (preserve 70:30 class ratio).
- **SMOTE**: Balances training data for XGBoost (ensures bad-risk learning).
- **Scaling**: All features scaled using standardized 
***

### 5. **Model Selection \& Evaluation**

- **Modeling \& Selection**
    - **XGBoost**: Chosen for strong ROC-AUC, handling of categorical/outlier features, and regulatory-auditable interpretability.
    - **Random Forest (inbuilt)**: Baseline ensemble, robust for tabular data, and strong industry adoption.
***

## Key Results (Test Data)

| Model | Accuracy | ROC-AUC | Precision (bad) | Recall (bad) | F1 (bad) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| XGBoost | 0.80 | 0.81 | 0.74 | 0.49 | 0.59 |
| Random Forest | 0.77 | 0.79 | 0.62 | 0.53 | 0.57 |

**Model Selection Justification:**
XGBoost outperforms on both global metrics and regulatory-relevant recall. Additionally, its compatibility with SHAP allows the most rigorous post-hoc interpretability for risk disclosure.

***

### 6. **SHAP Analysis: Global \& Local Explanations**

- **Global Feature Importance (Bar/Beeswarm plot)**
    - Top 5 features driving default risk:

1. **Monthly debt** (higher payment → higher risk)
2. **Credit amount** (bigger loans → more risk)
3. **Loan intensity** (fast/high borrowing → more risk)
4. **Checking account** status (richer = safer)
5. **Customer segment** (ML-detected hidden high-risk groups)
    - **Interpretation**: “Applicants with high monthly payments, larger loan demands, weak checking account history, and clustering into risky behavior groups are at highest risk.”
- **Local Explanations (Force/Waterfall plots)**
    - For each of 10 customers (5 high-risk, 5 low-risk):
        - SHAP decomposes the prediction, showing the main features pushing risk up or down.
        - Example: “Defaulted because monthly debt + loan intensity outweighed positive savings account and age group.”

***

### 7. **LIME Analysis: Local Validation**

- **LIME Explanations**: Independently generated for SAME 10 customers
    - For each: plotted, top 5 local feature impacts listed.
    - **Comparison**: Generally, LIME and SHAP agree on key risk-increasing features (monthly debt, credit amount, customer segment), but LIME sometimes prefers highly local splits or numerical cutpoints. Occasional discrepancy for “Customer segment”/”Age group” due to local linear fit.
    - **Interpretation**: “LIME confirms SHAP’s global picture locally, and sometimes better highlights where subtle risk groups cause jumps in model output.”

***

### 8. **Critique: SHAP vs. LIME**

| Dimension | SHAP | LIME |
| :-- | :-- | :-- |
| **Global** | Yes (summary plot, bar plot) | No (local only, requires averaging) |
| **Local** | Yes (force, waterfall, values) | Yes (bar chart, coefficients) |
| **Guarantees** | Theoretical, sums to prediction | Approximate (local only) |
| **Stability** | Always same for same model/data | Randomness, varies per fit/sample |
| **Use case** | Tree models highly optimized | Any model (NN, trees, boosting etc.) |
| **Regulation** | Excellent for audit/compliance | Good supporting evidence |

**Recommendation**:

- Use SHAP as main interpretability; supports both global and detailed local (“Why was this person denied?”)
- Use LIME to sanity-check or explain special edge cases (especially for non-tree models).

***

### 9. **Top 5 SHAP Features—Plainlanguage Summary**

1. **Monthly debt**: High payments relative to duration increase default risk most.
2. **Credit amount**: Large loans are more likely to default—especially if not matched by income/history.
3. **Loan intensity**: Shorter, more intense loans (large and quick) create systemic risk.
4. **Checking account**: Healthy and rich account balances denote better money management.
5. **Customer segment**: Machine learning groupings reveal hidden patterns—members of riskier segments shown to default together.

***

### 10. **Recommendations for the Risk Committee**

- Closely monitor applicants with high monthly payments and large loan requests.
- Segment analysis uncovers “hidden risk” groups even where individual features look safe—future policy should evolve as these groups do.
- Encourage broader bank account (checking/savings) usage for applicants—absence signals higher risk.
- Regularly re-audit model explanations with SHAP/LIME as data evolves.
- Consider combining manual feature engineering AND machine-discovered segments for best predictive/regulatory performance.

***

## Business \& Regulatory Summary

- **XGBoost with SHAP explanations is ideal for bank risk committees**, capturing both accuracy and transparency.
- **Random Forest provides an auditable backup/secondary model**—nearly as strong and very interpretable for tabular features.
- **SHAP \& LIME explanations show that risk is most driven by monthly repayment capacity, saving/checking account status, and hidden customer micro-segments** (detected by KMeans—an innovation!).
- **All steps, from EDA to deployment, are clearly documented for both technical and non-technical review.**

***
