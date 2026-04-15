# fda2_portfolio_app
Class project for Financial Data Analytics II at the University of Arkansas

## Assumptions & Methodology

This application applies mean–variance portfolio optimization to construct and evaluate portfolios using historical asset data. The results are driven by several key assumptions and modeling choices outlined below.

1. Return and Risk Estimation
Expected returns and the covariance matrix are estimated from historical price data using daily simple returns over a user-selected lookback window. These estimates are assumed to be representative of future performance. Portfolio risk is measured using standard deviation (volatility), and all return and risk metrics are annualized for comparability.

2. Mean–Variance Framework
Portfolio construction follows the mean–variance optimization framework, which assumes investors are risk-averse and seek to maximize expected return for a given level of risk. The efficient frontier represents the set of optimal portfolios under this framework, including the Global Minimum Variance (GMV) portfolio and the tangency (maximum Sharpe ratio) portfolio.

3. Risk-Free Rate and Capital Allocation
A user-specified risk-free rate is used to compute Sharpe ratios and construct the Capital Allocation Line (CAL). The model assumes investors can borrow and lend at this rate, which may not hold in real-world markets.

4. Portfolio Constraints
Portfolios are fully invested, meaning asset weights must sum to 1. The model includes a toggle for short selling, allowing users to permit or restrict negative weights. No additional weight constraints (such as upper or lower bounds on individual assets) are imposed.

5. Diversification and Correlation
Diversification benefits arise from imperfect correlations between assets. These correlations are estimated from historical data and assumed to remain stable over the investment horizon, though in practice they may change over time.

6. Sensitivity to Estimation Window
Mean-variance optimization is highly sensitive to its inputs, particularly expected returns and covariances. Because these inputs are estimated from historical data, even small changes in the selected lookback period can lead to significantly different portfolio weights and performance outcomes.

This sensitivity reflects the fact that asset returns and relationships are not stable over time. By comparing results across multiple estimation windows, the application illustrates how dependent optimized portfolios are on historical assumptions. As a result, these portfolios should be interpreted with caution, as they may not be robust or stable out of sample.

7. Limitations
This analysis does not account for transaction costs, taxes, liquidity constraints, or market frictions. It also assumes continuous rebalancing without cost and does not incorporate forward-looking information. Extreme market events or structural changes may not be fully captured by historical data.
