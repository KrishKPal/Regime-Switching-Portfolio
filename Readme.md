# Regime-Switching Portfolio

### HMM-Based Regime Detection × Convex Portfolio Optimization

A regime-aware portfolio construction framework combining a **walk-forward Hidden Markov Model (HMM)** with **regime-dependent convex optimization** and an **out-of-sample backtest with transaction costs**.

> **Pipeline:** Market Data → Feature Engineering → HMM Regime Detection → Convex Optimization → Transaction Costs → Out-of-Sample Evaluation

---

## Research Question

Can portfolio allocation adapt to changing market conditions by conditioning the portfolio's risk objective on the detected market regime?

Instead of maintaining a fixed allocation, the strategy dynamically changes its portfolio optimization objective according to the prevailing market state.

---

## Methodology

### Market Data

The portfolio uses four Indian-market proxies:

| Asset | Proxy |
|---|---|
| Equity | NIFTY 50 (`^NSEI`) |
| Gold | Gold BeES (`GOLDBEES.NS`) |
| Bonds | Gilt 5Y BeES (`GILT5YBEES.NS`) |
| Volatility | India VIX (`^INDIAVIX`) |

Data is retrieved using `yfinance` and aligned before modelling.

### Feature Engineering

The regime model uses market features designed to capture momentum and changing volatility conditions:

- 21-day momentum
- 21-day realized volatility
- India VIX
- Return-based features

Feature scaling is performed using training-window information only.

### Regime Detection

A **Gaussian Hidden Markov Model** is estimated using a walk-forward framework.

The model identifies a high-volatility **Crisis** state. The non-crisis state is further separated into **Bull** and **Bear** using momentum.

```text
Bull    → Positive momentum + non-crisis conditions
Bear    → Negative momentum + non-crisis conditions
Crisis  → High-volatility HMM state
```

This produces three economically interpretable market regimes.

---

## Walk-Forward Validation

The model is evaluated strictly out-of-sample.

At each walk-forward step:

1. Historical data is used to fit the feature scaler.
2. The HMM is fitted using only the training window.
3. The current market regime is classified.
4. Portfolio parameters are estimated from historical data.
5. Portfolio weights are optimized.
6. The next-period return is realized.
7. Transaction costs are applied according to portfolio turnover.

### Configuration

| Parameter | Value |
|---|---:|
| Minimum training window | 504 observations |
| Test window | 126 observations |
| Out-of-sample observations | 252 |
| Transaction cost | 7 bps |

---

## Regime-Dependent Portfolio Optimization

Portfolio weights are obtained using `CVXPY` under:

- Long-only constraints
- Fully invested portfolio
- No leverage

### Bull Regime

The optimizer balances expected return against portfolio variance:

$$
\max_w \quad \mu^T w - \lambda_{bull}w^T\Sigma w
$$

with:

$$
\lambda_{bull}=3
$$

### Bear Regime

A higher penalty is placed on portfolio variance:

$$
\max_w \quad \mu^T w - \lambda_{bear}w^T\Sigma w
$$

with:

$$
\lambda_{bear}=8
$$

### Crisis Regime

The strategy switches to minimum-variance allocation:

$$
\min_w \quad w^T\Sigma w
$$

subject to:

$$
w_i \geq 0,\qquad \sum_i w_i=1
$$

Expected returns are shrunk before optimization to reduce sensitivity to noisy historical estimates.

---

## Transaction Costs

The backtest incorporates proportional transaction costs of **7 basis points**.

Portfolio turnover is calculated as:

$$
Turnover_t=\sum_i |w_{i,t}-w_{i,t-1}|
$$

and transaction costs are:

$$
Cost_t=0.0007\times Turnover_t
$$

Both gross and net-of-cost returns are evaluated.

---

## Out-of-Sample Results

| Strategy | Sharpe | Sortino | Max Drawdown | Calmar | Annual Return |
|---|---:|---:|---:|---:|---:|
| Dynamic — Net | **3.03** | 4.81 | -2.67% | 3.93 | 10.52% |
| Dynamic — Gross | 3.14 | 4.96 | -2.67% | 4.07 | 10.88% |
| Equal Weight | 2.84 | **5.13** | **-2.55%** | **6.02** | 15.36% |
| 60/40 | 2.69 | 4.10 | -4.02% | 4.19 | **16.82%** |

**Annualized turnover of the dynamic strategy:** 5.09×

### Out-of-Sample Regime Distribution

| Regime | Observations |
|---|---:|
| Bull | 199 |
| Bear | 48 |
| Crisis | 5 |

The dynamic strategy produces the highest Sharpe ratio in the evaluated sample, while the static benchmarks achieve higher annualized returns. The results therefore illustrate a **risk-adjusted performance trade-off rather than universal dominance of regime switching**.

---

## Key Design Choices

- **Walk-forward HMM:** avoids using future observations during regime estimation.
- **Regime-dependent optimization:** changes portfolio risk preferences according to market conditions.
- **Return shrinkage:** reduces sensitivity to noisy expected-return estimates.
- **Transaction costs:** evaluates performance after portfolio turnover costs.
- **Static benchmarks:** provides a reference for interpreting the dynamic strategy.
- **Out-of-sample evaluation:** separates model estimation from performance measurement.

---

## Limitations

- Small asset universe
- Relatively short out-of-sample period
- HMM results depend on feature and state specification
- Expected-return estimates remain noisy
- Simplified transaction-cost model
- No explicit slippage or market-impact modelling
- No leverage or short selling
- Performance may be sensitive to lookback windows and hyperparameters

The reported metrics are **historical backtest results**, not evidence of future performance.

---

## Tech Stack

`Python` · `NumPy` · `Pandas` · `CVXPY` · `HMMlearn` · `yfinance` · `Matplotlib`

---

## Future Extensions

- Compare HMM with Markov-switching and change-point models
- Incorporate GARCH/EGARCH volatility forecasts
- Expand the asset universe
- Add covariance shrinkage
- Perform hyperparameter sensitivity analysis
- Test across multiple historical periods
- Add bootstrap confidence intervals
- Introduce purged and embargoed validation
- Jointly model portfolio turnover and transaction costs

---

*Research project in quantitative portfolio construction and systematic trading.*
