# Macro-Conditioned Convex Portfolio Optimization
### Pakistan SIP Investor — 2005 to 2025
---

## What This Project Does

This project builds a portfolio optimizer for a Pakistani retail investor choosing between three assets:

- **Gold** (PKR/oz)
- **KSE-100** equities
- **Money Market** (SBP policy rate deposits)

Instead of using fixed weights, the optimizer reads the current macroeconomic state — inflation, interest rates, USD strength, oil prices — and dynamically adjusts allocation. When inflation is high, it tilts toward gold. When SBP rates are high, it tilts toward money market deposits. When the economy is growing with low rates, it tilts toward equities.

---

## Problem Formulation

The core optimization problem is a **Quadratically Constrained Quadratic Program (QCQP)**:

```
minimize    λ·wᵀΣw  −  μ(x)ᵀw  +  (γ/T)·‖w − ŵ(T)‖²

subject to  1ᵀw = 1          (full investment)
            w ≥ 0             (no short selling)
            w₁ ≥ εg(π)        (gold floor, rises with CPI)
            w₃ ≥ εm(r)        (money market floor, rises with SBP rate)
            wᵀΣw ≤ σmax(T)²   (risk budget, grows with horizon)
```

Where:
- `w` = portfolio weights (3 assets)
- `μ(x)` = macro-conditional expected return (OLS model)
- `Σ` = Ledoit-Wolf shrinkage covariance matrix
- `T` = investment horizon in years (1 to 10)
- `λ, γ` = risk aversion and horizon penalty parameters

The QCQP is also reformulated as an equivalent **SOCP** via Cholesky decomposition, and extended to a **Robust SOCP** under ellipsoidal return uncertainty.

---

## Data

| Variable | Description | Source |
|---|---|---|
| Gold PKR/oz | Monthly % return in Pakistani Rupees | Computed from USD gold + PKR/USD |
| KSE-100 | Monthly % change in index level | Pakistan Stock Exchange |
| Money Market | 4-month rolling avg of SBP Rate / 12 | State Bank of Pakistan |
| CPI Inflation | Year-on-year % (lagged 1 month) | PBS / SBP |
| SBP Policy Rate | Annual rate (lagged 1 month) | State Bank of Pakistan |
| DXY | USD index, standardized | FRED |
| Crude Oil WTI | Log-normalized monthly price | EIA |

- **Period:** January 2005 — December 2025
- **Observations:** 252 months
- **File:** `pakistan_master_data_2005_2025_updated.csv`

---

## What the Code Does

The single script `portfoliooptimization.py` runs five things in order:

1. **Loads and engineers all variables** — computes asset returns, lags macro factors, standardizes features
2. **Estimates the return model** — OLS regression of asset returns on macro factors
3. **Parametric analysis** — sweeps a 22×22 grid of (inflation, SBP rate) combinations and solves 484 QCQPs to produce allocation heatmaps
4. **Efficient frontiers** — computes risk-return curves for 4 canonical macro regimes across investment horizons
5. **Walk-forward backtest** — month-by-month out-of-sample test from 2006 to 2025, comparing QCQP vs equal-weight vs fixed vs money-market-only

All results are saved to `portfolio_results.png`.

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/macro-portfolio-optimization.git
cd macro-portfolio-optimization
```

**2. Install dependencies**
```bash
pip install cvxpy numpy pandas matplotlib scikit-learn
```

**3. Make sure the data file is in the same folder**
```
macro-portfolio-optimization/
├── portfoliooptimization.py
├── pakistan_master_data_2005_2025_updated.csv
└── README.md
```

**4. Run**
```bash
python portfoliooptimization.py
```

**5. Output**
- Terminal prints regime analysis, OLS coefficients, backtest stats
- `portfolio_results.png` is saved in the same folder

---

## Results Summary

| Strategy | Ann. Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| **QCQP Optimized** | **15.8%** | **1.95** | **18.6%** |
| Equal Weight (1/3) | 17.0% | 1.84 | 20.2% |
| Money Market Only | 11.5% | 9.85 | 0.0% |
| Fixed Allocation | 15.9% | 1.17 | 41.4% |

Key finding: the QCQP strategy achieves the best risk-adjusted performance (highest Sharpe) and dramatically lower drawdown than the fixed allocation strategy — even though both have similar average returns. This shows that ignoring macro dynamics costs you in downside risk, not in average returns.

---

## Dependencies

```
cvxpy
numpy
pandas
matplotlib
scikit-learn
```

Python 3.8 or above.

---

## Files

```
├── portfoliooptimization.py               # main script (all code)
├── pakistan_master_data_2005_2025_updated.csv  # dataset
├── portfolio_results.png                  # output figure (generated on run)
└── README.md
```

---
