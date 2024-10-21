# Optimized Investment Portfolio Using Factor-Based Approach
## Project Overview
This project constructs an optimized investment portfolio using a factor-based approach to balance risk and return. The goal is to identify undervalued stocks with strong financial health and use data-driven methods to maximize returns while minimizing risk. This approach leverages financial metrics such as Price-to-Earnings (P/E) Ratio and Return on Equity (ROE) to select high-quality stocks, and applies various machine learning techniques like Principal Component Analysis (PCA) and K-Means Clustering to ensure diversification and reduce complexity. The final portfolio is optimized using Mean-Variance Optimization to achieve the best risk-return trade-off.

## Key Objectives:
#### Select High-Quality Stocks: Identify undervalued stocks based on their P/E Ratio and ROE to ensure better long-term returns.
#### Reduce Dimensionality with PCA: Simplify the stock return data to focus on key factors driving performance.
#### Ensure Diversification through K-Means Clustering: Group stocks into clusters based on return patterns to diversify and minimize overexposure to correlated assets.
#### Analyze Stock Relationships with Graphical Analysis: Visualize the correlations and dependencies between stocks to understand their interactions.
#### Optimize the Portfolio with Mean-Variance Optimization: Adjust stock weights to maximize the Sharpe Ratio, aiming to achieve the highest return for the lowest level of risk.
## Methodology
1. Stock Data Collection
Downloaded adjusted close price data for S&P 500 stocks from Yahoo Finance for the period between 2020 and 2023. The dataset is cleaned, and stocks with missing data are removed. This data provides the basis for calculating daily returns and conducting further analysis.

2. Fundamental Data Collection (P/E Ratio & ROE)
Using the P/E Ratio and Return on Equity (ROE), we assess the value and quality of the stocks. Stocks with lower P/E and higher ROE are considered more favorable, as they are both undervalued and exhibit strong financial health.

3. Stock Ranking
Based on the collected financial metrics, stocks are ranked according to their P/E (ascending) and ROE (descending). These ranks are combined, and the top 20 stocks with the best combined rank are selected for further analysis.

4. Principal Component Analysis (PCA)
PCA is applied to the daily returns of the selected stocks to reduce the dimensionality of the data. This step helps in focusing on the primary factors driving stock performance and simplifying the analysis. The explained variance for each principal component is plotted to understand the distribution of variance.

5. K-Means Clustering for Diversification
To ensure diversification, K-Means Clustering is performed on the stock returns. This groups the stocks into clusters based on their return patterns, helping avoid overexposure to any single sector or correlated assets. The clusters are visualized to demonstrate how stocks are grouped.

6. Graphical Analysis of Stock Relationships
We create a Correlation Matrix to visualize the relationships between stocks. Stocks with strong correlations are represented as edges in a Graphical Network Analysis, helping to understand how these stocks interact with each other. This analysis informs diversification strategies by highlighting correlations and dependencies between stocks.

7. Mean-Variance Optimization
Finally, we apply Mean-Variance Optimization to the portfolio to maximize the Sharpe Ratio. This step adjusts the weights of the selected stocks to achieve the best possible trade-off between risk and return. The optimized portfolio provides the highest return for the least amount of risk, based on historical data.

## Results
The optimized portfolio provides the following key results:

##### Optimized Portfolio Weights: A set of weights for each stock in the portfolio, calculated to maximize the risk-return trade-off.
##### Expected Annual Return: The projected annual return of the optimized portfolio.
##### Annual Volatility: The risk associated with the portfolio, represented as the standard deviation of portfolio returns.
##### Sharpe Ratio: A measure of risk-adjusted return, which indicates how much return is earned per unit of risk.


