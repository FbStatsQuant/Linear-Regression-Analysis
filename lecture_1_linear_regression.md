# Linear Regression

## Introduction

Linear regression is the foundation of supervised statistical learning. It models the relationship between a continuous outcome $y$ and a set of predictors $X$ by assuming that relationship is linear in the parameters. Despite its simplicity, it remains one of the most widely used models in applied statistics and machine learning — not because it is always correct, but because it is interpretable, computationally cheap, and a necessary baseline against which more complex models are measured.

The goal is not merely to fit a line. It is to estimate the conditional expectation $\mathbb{E}[y \mid X]$, quantify uncertainty around those estimates, and make valid inferences about which predictors matter and by how much. Each of these requires both a well-specified model and assumptions that hold in the data.

---

## 1. The Model

### Specification

Given $n$ observations and $p$ predictors, the linear regression model is:

$$y = X\beta + \varepsilon$$

where:

| Symbol | Dimension | Meaning |
|---|---|---|
| $y$ | $n \times 1$ | Response vector |
| $X$ | $n \times (p+1)$ | Design matrix (includes intercept column of ones) |
| $\beta$ | $(p+1) \times 1$ | Coefficient vector |
| $\varepsilon$ | $n \times 1$ | Error vector |

Written for a single observation $i$:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i$$

The intercept $\beta_0$ is included by appending a column of ones to $X$. Each $\beta_j$ represents the expected change in $y$ for a one-unit increase in $x_j$, holding all other predictors constant.

### Classical Assumptions

The validity of OLS estimates and their associated inference depends on the following assumptions:

| Assumption | Statement |
|---|---|
| **Linearity** | $\mathbb{E}[y \mid X] = X\beta$ |
| **Exogeneity** | $\mathbb{E}[\varepsilon \mid X] = 0$ |
| **Homoscedasticity** | $\mathrm{Var}(\varepsilon_i) = \sigma^2$ for all $i$ |
| **No autocorrelation** | $\mathrm{Cov}(\varepsilon_i, \varepsilon_j) = 0$ for $i \neq j$ |
| **No perfect multicollinearity** | $\mathrm{rank}(X) = p + 1$ |
| **Normality** | $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ (required for finite-sample inference) |

Together, assumptions 1–5 constitute the **Gauss-Markov conditions**, under which OLS is the Best Linear Unbiased Estimator (BLUE). Normality of errors is additionally required for exact t- and F-tests in small samples; asymptotically it is not needed.

---

## 2. Ordinary Least Squares

### The Objective

OLS finds $\hat{\beta}$ that minimizes the sum of squared residuals:

$$\hat{\beta} = \underset{\beta}{\arg\min} \sum_{i=1}^n (y_i - x_i^\top \beta)^2 = \underset{\beta}{\arg\min} \|y - X\beta\|^2$$

### Closed-Form Solution

Expanding $\|y - X\beta\|^2 = (y - X\beta)^\top(y - X\beta)$ and differentiating with respect to $\beta$:

$$\frac{\partial}{\partial \beta}\|y - X\beta\|^2 = -2X^\top(y - X\beta) = 0$$

Solving the **normal equations**:

$$X^\top X \hat{\beta} = X^\top y$$

$$\boxed{\hat{\beta} = (X^\top X)^{-1} X^\top y}$$

This requires $X^\top X$ to be invertible — i.e., no perfect multicollinearity among predictors.

### Geometric Interpretation

The OLS fit $\hat{y} = X\hat{\beta}$ is the orthogonal projection of $y$ onto the column space of $X$:

$$\hat{y} = X(X^\top X)^{-1}X^\top y = Hy$$

where $H = X(X^\top X)^{-1}X^\top$ is the **hat matrix** (projection matrix). The residuals $e = y - \hat{y} = (I - H)y$ are orthogonal to $\hat{y}$:

$$\hat{y}^\top e = 0$$

This orthogonality is not an assumption — it is an algebraic consequence of OLS.

---

## 3. Properties of the OLS Estimator

### Unbiasedness

Under the assumption $\mathbb{E}[\varepsilon \mid X] = 0$:

$$\mathbb{E}[\hat{\beta}] = \mathbb{E}[(X^\top X)^{-1}X^\top y] = \beta + (X^\top X)^{-1}X^\top\mathbb{E}[\varepsilon] = \beta$$

OLS is unbiased — on average, it recovers the true $\beta$.

### Variance

Under homoscedasticity $\mathrm{Var}(\varepsilon) = \sigma^2 I$:

$$\mathrm{Var}(\hat{\beta}) = \sigma^2 (X^\top X)^{-1}$$

The diagonal elements of this matrix are the variances of each coefficient. Their square roots are the **standard errors** reported in regression output. In practice $\sigma^2$ is estimated by:

$$\hat{\sigma}^2 = \frac{\|y - X\hat{\beta}\|^2}{n - p - 1} = \frac{\mathrm{RSS}}{n - p - 1}$$

where the denominator corrects for the $p+1$ estimated parameters.

### Gauss-Markov Theorem

Among all linear unbiased estimators, OLS has the smallest variance. Formally, for any other linear unbiased estimator $\tilde{\beta}$:

$$\mathrm{Var}(\tilde{\beta}) - \mathrm{Var}(\hat{\beta}) \succeq 0$$

This is the sense in which OLS is **BLUE** — Best Linear Unbiased Estimator.

---

## 4. Inference

### t-Tests for Individual Coefficients

Under the null hypothesis $H_0: \beta_j = 0$:

$$t_j = \frac{\hat{\beta}_j}{\mathrm{SE}(\hat{\beta}_j)} \sim t_{n-p-1}$$

where $\mathrm{SE}(\hat{\beta}_j) = \hat{\sigma}\sqrt{[(X^\top X)^{-1}]_{jj}}$. A small p-value indicates that the predictor carries information about $y$ beyond what the other predictors already explain.

### F-Test for Overall Significance

Tests whether any predictor is useful, i.e., $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$:

$$F = \frac{(\mathrm{TSS} - \mathrm{RSS})/p}{\mathrm{RSS}/(n-p-1)} \sim F_{p,\, n-p-1}$$

where $\mathrm{TSS} = \sum_i (y_i - \bar{y})^2$ is the total sum of squares.

### Coefficient of Determination

$R^2$ measures the proportion of variance in $y$ explained by the model:

$$R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}} = \frac{\mathrm{ESS}}{\mathrm{TSS}}$$

$R^2$ always increases when predictors are added, even irrelevant ones. The **adjusted** $R^2$ penalizes for model complexity:

$$\bar{R}^2 = 1 - \frac{\mathrm{RSS}/(n-p-1)}{\mathrm{TSS}/(n-1)}$$

---

## 5. Heteroscedasticity and Robust Standard Errors

### The Problem

When $\mathrm{Var}(\varepsilon_i) = \sigma_i^2$ varies across observations (heteroscedasticity), the standard OLS variance formula $\sigma^2(X^\top X)^{-1}$ is wrong. Standard errors are biased, t-statistics are invalid, and inference cannot be trusted — even though $\hat{\beta}$ itself remains unbiased.

### HC3 Robust Standard Errors

Instead of assuming constant variance, **Heteroscedasticity-Consistent (HC)** estimators estimate the variance-covariance matrix of $\hat{\beta}$ directly from the residuals. The HC3 variant is:

$$
\widehat{\mathrm{Var}}_{\mathrm{HC3}}(\hat{\beta}) = (X^\top X)^{-1} \left(\sum_{i=1}^n \frac{e_i^2}{(1-h_{ii})^2} x_i x_i^\top \right) (X^\top X)^{-1}
$$

where $h_{ii}$ is the leverage of observation $i$ — the $i$-th diagonal of the hat matrix $H = X(X^\top X)^{-1}X^\top$.

The $(1-h_{ii})^2$ term in the denominator corrects for the bias that arises when high-leverage points are present. HC3 is the most conservative and recommended variant in moderate samples.

**Key point**: HC3 does not change $\hat{\beta}$, $\hat{y}$, or residuals. It only changes the standard errors and hence all downstream inference. The model fit is identical.

---

## 6. Multicollinearity

### The Problem

When predictors are highly correlated, $X^\top X$ approaches singularity. Even before exact singularity, near-multicollinearity inflates $\mathrm{Var}(\hat{\beta}) = \sigma^2(X^\top X)^{-1}$: coefficients become highly unstable, standard errors balloon, and individual t-tests lose power. Adding or removing a single observation can swing coefficient estimates dramatically.

### Variance Inflation Factor

The **Variance Inflation Factor** for predictor $j$ quantifies how much its variance is inflated by collinearity:

$$\mathrm{VIF}_j = \frac{1}{1 - R^2_j}$$

where $R^2_j$ is the $R^2$ from regressing $x_j$ on all other predictors. Interpretation:

| VIF | Interpretation |
|---|---|
| $1$ | No collinearity |
| $1$–$5$ | Mild, generally acceptable |
| $5$–$10$ | Moderate concern |
| $> 10$ | Severe — coefficient estimates unreliable |

### Remedies

The simplest remedy is variable removal — drop one of the offending correlated predictors, retaining whichever has stronger theoretical justification or predictive value. Alternatively, **Ridge regression** addresses multicollinearity by adding an $L_2$ penalty to the OLS objective:

$$\hat{\beta}_{\mathrm{ridge}} = \underset{\beta}{\arg\min} \|y - X\beta\|^2 + \lambda\|\beta\|^2 = (X^\top X + \lambda I)^{-1}X^\top y$$

Adding $\lambda I$ to $X^\top X$ guarantees invertibility regardless of collinearity, at the cost of introducing bias.

---

## 7. Implementation in Python

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import statsmodels.api as sm

# Load data
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Drop high-VIF predictors
X = X.drop(columns=['s1', 's2'])

# Add intercept and fit with HC3 robust standard errors
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit(cov_type='HC3')

print(model.summary())
```

The `statsmodels` summary reports: coefficient estimates, HC3 standard errors, t-statistics, p-values, confidence intervals, $R^2$, adjusted $R^2$, and the F-statistic. It is the primary output to interpret.

---

## 8. Strengths and Weaknesses

### Strengths

**Interpretability**: Each coefficient has a direct, quantitative interpretation — a one-unit change in $x_j$ produces a $\hat{\beta}_j$ change in $\hat{y}$, all else equal. No other model of comparable predictive power offers this clarity.

**Efficiency**: Under Gauss-Markov conditions, OLS is the minimum-variance unbiased estimator in the class of linear estimators. For well-specified problems, you cannot do better without introducing bias.

**Closed-form solution**: Unlike iterative methods, OLS has an exact analytical solution. No hyperparameters, no convergence criteria, no local minima.

**Robustness of inference**: With HC3 standard errors, valid inference is achievable even when homoscedasticity fails — a common condition in practice.

### Weaknesses

**Linearity constraint**: The model cannot capture nonlinear relationships without manual feature engineering. If the true $\mathbb{E}[y \mid X]$ is nonlinear, OLS is misspecified.

**Sensitivity to outliers**: The squared loss amplifies the influence of extreme observations. A single high-leverage, high-residual point can substantially distort $\hat{\beta}$.

**Assumption dependence**: Valid inference requires the Gauss-Markov conditions. Violations — particularly heteroscedasticity and multicollinearity — must be detected and addressed explicitly.

**No regularization**: Standard OLS does not penalize model complexity. In high-dimensional settings ($p \approx n$), it overfits and becomes unstable.

---

## 9. Summary

1. **The linear model** $y = X\beta + \varepsilon$ estimates $\mathbb{E}[y \mid X]$ as a linear function of predictors. The coefficient $\beta_j$ is the partial effect of $x_j$ on $y$ holding other predictors fixed.

2. **OLS minimizes** $\|y - X\beta\|^2$, yielding the closed-form solution $\hat{\beta} = (X^\top X)^{-1}X^\top y$. Under Gauss-Markov conditions it is unbiased and minimum-variance.

3. **Inference** relies on standard errors derived from $\mathrm{Var}(\hat{\beta}) = \sigma^2(X^\top X)^{-1}$. When homoscedasticity fails, these standard errors are incorrect and HC3 robust standard errors must be used instead.

4. **Multicollinearity** inflates coefficient variances and makes individual estimates unstable. VIF scores above 10 signal a problem; the remedy is variable removal or Ridge regularization.

5. **HC3 robust standard errors** correct for heteroscedasticity without changing the fitted values or coefficients — only standard errors and downstream inference are affected.

6. **Diagnostics are not optional.** The model's validity depends on its assumptions holding, and those assumptions must be actively checked in every analysis.

---

## References

- Greene, W. H. (2012). *Econometric Analysis* (7th ed.). Pearson.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.

---

**Next Lecture**: Linear Regression Diagnostics
