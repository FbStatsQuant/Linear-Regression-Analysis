# Linear Regression Analysis

Rigorous exploration of Linear Regression in Python and R, including data simulation, model fitting, and assumption testing (linearity, normality, homoscedasticity, independence, multicollinearity). Combines theory with practice for a complete understanding of regression analysis.

---

## 1. Project Goals

This repository is meant to be a **complete, hands-on lab for linear regression**, not just a collection of code snippets. The focus is on

- constructing controlled synthetic data where the true data–generating process is known,
- fitting linear regression models in both **Python** and **R**,
- **systematically checking and stressing each classical assumption**, and
- quantifying how violations of those assumptions affect estimation, inference, and prediction accuracy.

The target audience is students and practitioners who already know the basic theory and want to see it exercised carefully on real code and real (or realistically simulated) data.

---

## 2. Overview of the Analysis Workflow

Each example in this repository follows roughly the same pipeline:

1. **Data generation / loading**
   - Synthetic or real datasets for demonstration.
   - Mix of continuous and categorical predictors.
   - Controlled noise structure (Gaussian, non-Gaussian, heteroscedastic, correlated errors, etc.).

2. **Exploratory Data Analysis (EDA)**
   - Summary statistics and correlation structure.
   - Visual inspection (scatterplots, boxplots, pair plots, etc.).
   - Basic feature engineering for categorical variables (dummy / one-hot encoding).

3. **Model specification and fitting**
   - Baseline **Ordinary Least Squares (OLS)** model.
   - Alternative specifications (interaction terms, transformations, polynomial terms, etc.).
   - Parallel implementations in **Python** and **R**.

4. **Assumption checking and diagnostics**
   - Linearity and functional form.
   - Independence and autocorrelation.
   - Homoscedasticity vs. heteroscedasticity.
   - Normality of errors.
   - Multicollinearity.
   - Leverage, influence, and outliers.

5. **Model evaluation**
   - In-sample diagnostics and residual analysis.
   - Train/test splits or cross-validation.
   - Comparison of estimated coefficients to true parameters (for synthetic data).
   - Predictive performance metrics (RMSE, MAE, \(R^2\), adjusted \(R^2\)).

6. **Reporting and interpretation**
   - Interpretation of coefficients, confidence intervals, and p-values.
   - Discussion of what breaks when assumptions fail.
   - Comparison between Python and R workflows.

---

## 3. Assumptions and Diagnostics

For each fitted model, we explicitly check:

- **Linearity**
  - Residuals vs. fitted plots.
  - Partial residual (component+residual) plots.
  - Comparison with models including nonlinear terms (e.g., polynomials, splines).

- **Independence**
  - Residual autocorrelation plots (ACF/PACF) when data are ordered.
  - Durbin–Watson and related tests in examples with time structure.

- **Homoscedasticity**
  - Residuals vs. fitted values and vs. predictors.
  - Scale–location plots.
  - Formal tests such as Breusch–Pagan / White tests (when appropriate).

- **Normality of errors**
  - Q–Q plots of residuals.
  - Histograms and kernel density estimates.
  - Tests such as Shapiro–Wilk (with the usual caveats).

- **Multicollinearity**
  - Correlation matrix and heatmaps.
  - Variance Inflation Factors (VIFs).
  - Condition numbers for the design matrix.

- **Leverage and influence**
  - Leverage vs. residual plots.
  - Cook’s distance, DFBETAs, and related measures.

---

## 4. Repository Structure

Planned layout (may evolve as the project grows):

```text
Linear-Regression-Analysis-/
├─ python/
│  ├─ notebooks/          # Jupyter notebooks with step-by-step analysis
│  └─ scripts/            # Reusable Python scripts (data gen, models, plots)
├─ r/
│  ├─ scripts/            # R scripts and RMarkdown analyses
│  └─ notebooks/          # Optional: .Rmd or Quarto documents
├─ data/
│  ├─ raw/                # Original data (if any real datasets are used)
│  └─ processed/          # Cleaned / simulated datasets ready for modeling
├─ figures/               # Diagnostic plots and summary graphics
├─ docs/                  # Additional documentation, notes, or references
└─ README.md              # This file
