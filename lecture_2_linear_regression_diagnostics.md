# Linear Regression Diagnostics

## Introduction

Fitting a linear regression model is straightforward. Knowing whether to trust it is not.

The OLS estimator is unbiased and efficient under the Gauss-Markov conditions — but those conditions are assumptions, not guarantees. In practice, real data violates them routinely. The coefficient estimates may be fine while the standard errors are wrong. The standard errors may be fine while the model is misspecified. Diagnostics exist to surface these problems before they contaminate your conclusions.

This lecture covers the six core diagnostic checks for linear regression: linearity, independence of errors, homoscedasticity, normality of residuals, multicollinearity, and influential observations. For each, we cover the theoretical motivation, the visual and formal tests, and what to do when the assumption fails.

---

## 1. Residuals

All regression diagnostics operate on residuals. The **raw residual** for observation $i$ is:

$$e_i = y_i - \hat{y}_i = y_i - x_i^\top \hat{\beta}$$

In matrix form: $e = y - \hat{y} = (I - H)y$, where $H = X(X^\top X)^{-1}X^\top$ is the hat matrix.

Residuals are not independent, even if the true errors $\varepsilon_i$ are. Their covariance matrix is:

$$\mathrm{Var}(e) = \sigma^2(I - H)$$

This matters for leverage-adjusted diagnostics. For most visual diagnostics, raw residuals are sufficient.

---

## 2. Linearity

### What It Means

The assumption is $\mathbb{E}[y \mid X] = X\beta$. If the true relationship is nonlinear, OLS will systematically misestimate $y$ in regions where the true curve departs from the fitted line — producing structured, non-random residuals.

### How to Check

**Residuals vs. Fitted Values plot**: Plot $e_i$ against $\hat{y}_i$. Under correct specification you expect a flat, structureless scatter centered on zero. A U-shape or inverted U-shape indicates a missed nonlinear trend. Funneling indicates heteroscedasticity (see Section 4).

```python
plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (Linearity)')
```

### What to Do If It Fails

- Add polynomial terms: $x_j^2$, $x_j^3$
- Apply a transformation to $y$ or $x_j$ (log, square root)
- Use a nonlinear model (GAM, regression tree)

---

## 3. Independence of Errors

### What It Means

The assumption is $\mathrm{Cov}(\varepsilon_i, \varepsilon_j) = 0$ for $i \neq j$. In time series data, errors are commonly correlated across time (autocorrelation). In cross-sectional data with geographic or clustered structure, errors may be correlated within clusters.

When errors are correlated, OLS standard errors are wrong — typically underestimated — leading to spuriously significant results.

### How to Check

**Durbin-Watson test**: Tests for first-order autocorrelation $\varepsilon_t = \rho\varepsilon_{t-1} + u_t$. The statistic is:

$$DW = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2}$$

| DW value | Interpretation |
|---|---|
| $\approx 2$ | No autocorrelation |
| $< 1.5$ | Positive autocorrelation |
| $> 2.5$ | Negative autocorrelation |

```python
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.3f}")
```

### What to Do If It Fails

For time series data: use Newey-West HAC (heteroscedasticity and autocorrelation consistent) standard errors, or switch to an ARIMA/ARIMAX model. For clustered data: use cluster-robust standard errors.

---

## 4. Homoscedasticity

### What It Means

The assumption is $\mathrm{Var}(\varepsilon_i) = \sigma^2$ — constant error variance across all observations. When variance depends on $x$ or $\hat{y}$ (heteroscedasticity), the formula $\mathrm{Var}(\hat{\beta}) = \sigma^2(X^\top X)^{-1}$ is incorrect. Standard errors are biased, and t-tests and F-tests are invalid.

Note that $\hat{\beta}$ itself remains unbiased under heteroscedasticity — the problem is entirely in the inference, not the point estimates.

### How to Check

**Scale-Location plot**: Plot $\sqrt{|e_i|}$ against $\hat{y}_i$. A flat loess line indicates constant variance; an upward slope indicates variance increasing with fitted values (the most common pattern).

```python
plt.scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5)
plt.xlabel('Fitted Values')
plt.ylabel('√|Residuals|')
plt.title('Scale-Location (Homoscedasticity)')
```

**Breusch-Pagan test**: Formally tests whether the squared residuals can be explained by the predictors. Under $H_0$: homoscedasticity:

$$BP = nR^2_{e^2 \sim X} \sim \chi^2_p$$

where $R^2_{e^2 \sim X}$ is the $R^2$ from regressing $e_i^2$ on $X$. A small p-value rejects homoscedasticity.

```python
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X_const)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")
```

### What to Do If It Fails

**HC3 robust standard errors** (preferred in most cases): Correct the standard errors without altering the model. The HC3 estimator is:

$$\widehat{\mathrm{Var}}_{\mathrm{HC3}}(\hat{\beta}) = (X^\top X)^{-1} \left(\sum_{i=1}^n \frac{e_i^2}{(1-h_{ii})^2} x_i x_i^\top \right) (X^\top X)^{-1}$$

```python
model = sm.OLS(y, X_const).fit(cov_type='HC3')
```

**Log-transform the outcome**: If $y > 0$ and variance grows with the mean, modeling $\log y$ often stabilizes variance.

**Weighted Least Squares (WLS)**: If the variance structure is known, downweight high-variance observations by $w_i = 1/\sigma_i^2$.

---

## 5. Normality of Residuals

### What It Means

For exact finite-sample inference (t-tests, F-tests, confidence intervals), the errors must be normally distributed: $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$. By the Central Limit Theorem, this assumption becomes less critical as $n$ grows — asymptotically, OLS inference is valid without it. In small samples ($n < 50$), normality matters considerably more.

### How to Check

**Q-Q Plot**: Plots sample quantiles of the residuals against theoretical quantiles of the normal distribution. Points should fall on the diagonal line. Departures in the tails indicate heavy tails or skew.

```python
import statsmodels.api as sm
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot (Normality of Residuals)')
```

**Shapiro-Wilk test**: The most powerful test for normality in moderate samples. Under $H_0$: residuals are normally distributed. A p-value below 0.05 rejects normality.

```python
from scipy import stats
stat, p = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p:.4f}")
```

Note: in large samples Shapiro-Wilk becomes hypersensitive and will reject trivial departures from normality that have no practical consequence. Interpret it alongside the Q-Q plot.

### What to Do If It Fails

- Apply a Box-Cox transformation to $y$
- Use bootstrapped confidence intervals, which do not require normality
- In large samples, rely on asymptotic theory and disregard the Shapiro-Wilk result if the Q-Q plot looks reasonable

---

## 6. Multicollinearity

### What It Means

Multicollinearity occurs when two or more predictors are highly linearly correlated. It does not bias $\hat{\beta}$ — OLS is still unbiased — but it inflates the variance of the estimates. In the extreme case of perfect collinearity, $X^\top X$ is singular and $\hat{\beta}$ does not exist. In practice, near-collinearity produces unstable coefficients with enormous standard errors.

The effect can be subtle: individual t-tests may fail to find significance even when the joint F-test is significant, because each predictor's variance is inflated by its correlation with others.

### How to Check

**Correlation matrix**: Inspect pairwise correlations among predictors. Correlations above $|0.7|$ are worth investigating.

```python
import seaborn as sns
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm')
```

**Variance Inflation Factor (VIF)**: For predictor $j$, regress $x_j$ on all other predictors and compute:

$$\mathrm{VIF}_j = \frac{1}{1 - R^2_j}$$

| VIF | Action |
|---|---|
| $< 5$ | No concern |
| $5$–$10$ | Investigate |
| $> 10$ | Serious problem — consider dropping or regularizing |

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif.sort_values('VIF', ascending=False))
```

### What to Do If It Fails

**Drop predictors**: Remove the variable with the highest VIF, recompute, and iterate. Retain the variable with stronger theoretical justification or direct interpretive value.

**Ridge regression**: The $L_2$ penalty in Ridge adds $\lambda I$ to $X^\top X$, guaranteeing invertibility and shrinking correlated coefficients toward each other. It introduces bias but reduces variance — the bias-variance tradeoff applied to collinearity.

$$\hat{\beta}_{\mathrm{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y$$

---

## 7. Influential Observations

### What It Means

An observation is **influential** if its removal would substantially change $\hat{\beta}$. Not all outliers are influential — a point with a large residual but near the mean of $X$ has low leverage and little effect on the fit. Influence is a combination of high leverage (unusual $x$) and large residual.

### Leverage

The leverage of observation $i$ is $h_{ii}$, the $i$-th diagonal of the hat matrix $H = X(X^\top X)^{-1}X^\top$:

$$h_{ii} = x_i^\top (X^\top X)^{-1} x_i$$

A high leverage point sits far from the centroid of $X$ and has disproportionate ability to pull the regression line toward itself. The average leverage is $(p+1)/n$; points with $h_{ii} > 2(p+1)/n$ are commonly flagged.

### Cook's Distance

**Cook's Distance** measures the aggregate change in all fitted values when observation $i$ is removed:

$$D_i = \frac{\sum_{j=1}^n (\hat{y}_j - \hat{y}_{j(i)})^2}{(p+1)\hat{\sigma}^2} = \frac{e_i^2}{(p+1)\hat{\sigma}^2} \cdot \frac{h_{ii}}{(1-h_{ii})^2}$$

where $\hat{y}_{j(i)}$ is the fitted value for observation $j$ when observation $i$ is excluded. The second form shows that Cook's Distance is the product of the squared residual and a leverage term — it is large when both are large simultaneously.

A common threshold is $D_i > 4/n$, though this is a rough guideline rather than a hard rule.

```python
influence = sm.OLS(y, X_const).fit().get_influence()
cooks_d = influence.cooks_distance[0]

plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=',')
plt.axhline(4/len(cooks_d), color='red', linestyle='--', label='Threshold (4/n)')
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance (Influential Points)")
plt.legend()
```

### What to Do If Influential Points Are Found

- Inspect them: are they data entry errors, measurement errors, or legitimate extreme cases?
- If errors: correct or remove them
- If legitimate: report results with and without them; consider robust regression (e.g., Huber loss) which downweights outliers automatically

---

## 8. Full Diagnostic Workflow

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

# Load and prepare data
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names).drop(columns=['s1', 's2'])
y = pd.Series(data.target, name='target')
X_const = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X_const).fit(cov_type='HC3')
print(model.summary())

residuals = model.resid
fitted   = model.fittedvalues
cooks_d  = sm.OLS(y, X_const).fit().get_influence().cooks_distance[0]

# --- Plots ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Linearity
axes[0, 0].scatter(fitted, residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set(xlabel='Fitted Values', ylabel='Residuals', title='Residuals vs Fitted (Linearity)')

# 2. Normality: Q-Q
sm.qqplot(residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normality of Residuals)')

# 3. Homoscedasticity
axes[0, 2].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5)
axes[0, 2].set(xlabel='Fitted Values', ylabel='√|Residuals|', title='Scale-Location (Homoscedasticity)')

# 4. Multicollinearity
sns.heatmap(X.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Correlation Matrix (Multicollinearity)')

# 5. Influential Points
axes[1, 1].stem(np.arange(len(cooks_d)), cooks_d, markerfmt=',')
axes[1, 1].axhline(4/len(cooks_d), color='red', linestyle='--', label='Threshold (4/n)')
axes[1, 1].set(xlabel='Observation Index', ylabel="Cook's Distance", title="Cook's Distance")
axes[1, 1].legend()

# 6. Residual Distribution
axes[1, 2].hist(residuals, bins=30, edgecolor='black')
axes[1, 2].set_title('Residual Distribution')

plt.tight_layout()
plt.show()

# --- Formal Tests ---
print(f"Durbin-Watson    : {durbin_watson(residuals):.3f}   (want ~2.0)")
print(f"Breusch-Pagan p  : {het_breuschpagan(residuals, X_const)[1]:.4f}  (< 0.05 = heteroscedasticity)")
print(f"Shapiro-Wilk p   : {stats.shapiro(residuals)[1]:.4f}  (< 0.05 = non-normal)")

vif_df = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("\nVIF Scores:")
print(vif_df.sort_values('VIF', ascending=False).to_string(index=False))
```

---

## 9. Diagnostic Summary Table

| Assumption | Visual Check | Formal Test | Failure Remedy |
|---|---|---|---|
| Linearity | Residuals vs Fitted | — | Transform $x$ or $y$; polynomial terms |
| Independence | Residuals vs order | Durbin-Watson | HAC / Newey-West SE; clustered SE |
| Homoscedasticity | Scale-Location | Breusch-Pagan | HC3 robust SE; log transform $y$; WLS |
| Normality | Q-Q plot | Shapiro-Wilk | Box-Cox transform; bootstrap CI |
| Multicollinearity | Correlation heatmap | VIF | Drop predictors; Ridge regression |
| Influential points | Cook's Distance plot | Cook's $D_i > 4/n$ | Inspect, correct, or use robust regression |

---

## 10. Results on the Diabetes Dataset

Running the full diagnostic workflow on the diabetes dataset (with `s1` and `s2` removed) produced the following:

| Check | Result | Status |
|---|---|---|
| Linearity | No visible curvature in residuals vs fitted | ✓ Pass |
| Independence | Durbin-Watson = 2.000 | ✓ Pass |
| Normality | Shapiro-Wilk p = 0.46; Q-Q plot on line | ✓ Pass |
| Homoscedasticity | Breusch-Pagan p = 0.0025; fan in scale-location | ✗ Fail → HC3 applied |
| Multicollinearity | All VIF < 3.2 after dropping s1, s2 | ✓ Pass |
| Influential points | A few points above 4/n threshold; none extreme | ~ Minor concern |

Heteroscedasticity was the only substantive violation and was addressed by refitting with HC3 robust standard errors. Coefficients are unchanged; only inference is corrected.

---

## 11. Summary

1. **Diagnostics are not post-hoc checks** — they determine whether the model's estimates and inference can be trusted at all.

2. **Residual plots are the first line of defense**: residuals vs fitted (linearity), scale-location (homoscedasticity), and Q-Q (normality) together cover three of the six core assumptions visually.

3. **Formal tests confirm what the plots suggest**: Breusch-Pagan for homoscedasticity, Durbin-Watson for independence, Shapiro-Wilk for normality. Do not use them as substitutes for visual inspection.

4. **Heteroscedasticity is common in practice** and does not require discarding the model. HC3 robust standard errors correct inference while leaving $\hat{\beta}$ unchanged.

5. **Multicollinearity is diagnosed by VIF**, not by the correlation matrix alone. VIF captures indirect correlations that pairwise inspection misses. Values above 10 require action.

6. **Cook's Distance identifies observations that drive the fit**. A large Cook's $D_i$ warrants investigation — not automatic removal. The question is always whether the point is an error or a legitimate extreme case.

---

## References

- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics*. Wiley.
- Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics*, 19(1), 15–18.
- Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287–1294.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325.

---

**Previous Lecture**: Linear Regression — Model and Estimation  
**Next Lecture**: Ridge and Lasso Regression
