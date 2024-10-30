import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.optimize import minimize

# Step 1: Download S&P 500 Stock Data
def download_stock_data(start_date, end_date):
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    stocks_data = yf.download(sp500_tickers, start=start_date, end=end_date)['Adj Close']
    return stocks_data.dropna(axis=1)  # Remove columns with missing data

# Step 2: Get P/E Ratio and ROE data
def get_fundamental_data(tickers):
    pe_ratios = {}
    roe = {}

    # Fetch P/E and ROE data for each stock
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            pe_ratios[ticker] = stock.info.get('trailingPE')
            roe[ticker] = stock.info.get('returnOnEquity')
        except:
            continue

    pe_ratios_df = pd.DataFrame.from_dict(pe_ratios, orient='index', columns=['PE_Ratio']).dropna()
    roe_df = pd.DataFrame.from_dict(roe, orient='index', columns=['ROE']).dropna()
    
    return pe_ratios_df, roe_df

# Step 3: Rank stocks based on P/E and ROE
def rank_stocks(pe_ratios_df, roe_df, top_n=20):
    # Merge P/E and ROE DataFrames
    fundamentals = pe_ratios_df.join(roe_df, how='inner')
    
    # Convert P/E Ratio and ROE to numeric, coercing errors to NaN
    fundamentals['PE_Ratio'] = pd.to_numeric(fundamentals['PE_Ratio'], errors='coerce')
    fundamentals['ROE'] = pd.to_numeric(fundamentals['ROE'], errors='coerce')
    
    # Drop rows with missing data
    fundamentals = fundamentals.dropna(subset=['PE_Ratio', 'ROE'])
    
    # Rank stocks
    fundamentals['PE_Rank'] = fundamentals['PE_Ratio'].rank(ascending=True)
    fundamentals['ROE_Rank'] = fundamentals['ROE'].rank(ascending=False)
    fundamentals['Combined_Rank'] = fundamentals['PE_Rank'] + fundamentals['ROE_Rank']
    
    # Select top stocks based on combined rank
    return fundamentals.nsmallest(top_n, 'Combined_Rank').index.tolist()

# Step 4: Calculate daily returns for the selected stocks
def get_stock_returns(stocks_data, selected_stocks):
    top_20_data = stocks_data[selected_stocks]
    return top_20_data.pct_change().dropna()

# Step 5: Perform PCA
def perform_pca(returns):
    pca = PCA()
    pca.fit(returns)
    explained_variance = pca.explained_variance_ratio_
    
    # Plot explained variance by components
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100,
            tick_label=[f'PC{i}' for i in range(1, len(explained_variance) + 1)])
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained (%)')
    plt.show()
    
    return pca

# Step 6: Apply K-Means clustering
def apply_kmeans(returns, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(returns)
    returns['Cluster'] = clusters
    
    # Plot clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=returns.iloc[:, 0], y=returns.iloc[:, 1], hue=clusters, palette='Set1', s=100)
    plt.title('K-Means Clustering of Selected Stocks')
    plt.show()
    
    return clusters

# Step 7: Perform correlation and graphical analysis
def plot_correlation_graph(returns):
    correlation_matrix = returns.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Selected Stocks')
    plt.show()
    
    # Create a graph based on correlations
    G = nx.Graph()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.6:  # Correlation threshold
                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j])

    # Plot the graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', edge_color='grey', node_size=500, font_size=10)
    plt.title('Graphical Analysis of Stock Correlations')
    plt.show()

# Step 8: Optimize the portfolio using Mean-Variance Optimization
def optimize_portfolio(returns):
    returns_for_optimization = returns.drop(columns=['Cluster'])
    
    # Function to calculate portfolio performance
    def portfolio_performance(weights, returns):
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_stddev
        return portfolio_return, portfolio_stddev, sharpe_ratio

    # Function to minimize negative Sharpe Ratio
    def minimize_sharpe(weights, returns):
        return -portfolio_performance(weights, returns)[2]

    # Initial guess (equal weights)
    initial_weights = np.ones(len(returns_for_optimization.columns)) / len(returns_for_optimization.columns)
    
    # Bounds and constraints
    bounds = tuple((0, 1) for _ in range(len(returns_for_optimization.columns)))
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    # Optimize using SLSQP
    optimized_result = minimize(minimize_sharpe, initial_weights, args=(returns_for_optimization,), 
                                method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimized_weights = optimized_result.x
    
    # Display optimized portfolio
    portfolio = pd.DataFrame({'Stock': returns_for_optimization.columns, 'Weight': optimized_weights})
    print("Optimized Portfolio Weights:")
    print(portfolio)
    
    portfolio_return, portfolio_stddev, sharpe_ratio = portfolio_performance(optimized_weights, returns_for_optimization)
    print(f"Expected Annual Return: {portfolio_return:.2%}")
    print(f"Annual Volatility: {portfolio_stddev:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return portfolio

# Main program
if __name__ == "__main__":
    # Define time period
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Step 1: Download stock data
    stocks_data = download_stock_data(start_date, end_date)

    # Step 2: Get fundamental data
    tickers = stocks_data.columns.tolist()
    pe_ratios_df, roe_df = get_fundamental_data(tickers)

    # Step 3: Rank stocks and select top 20
    top_20_stocks = rank_stocks(pe_ratios_df, roe_df, top_n=20)

    # Step 4: Get stock returns for top 20 stocks
    returns = get_stock_returns(stocks_data, top_20_stocks)

    # Step 5: Perform PCA
    pca = perform_pca(returns)

    # Step 6: Apply K-Means clustering
    clusters = apply_kmeans(returns)

    # Step 7: Perform correlation and graphical analysis
    plot_correlation_graph(returns)

    # Step 8: Optimize the portfolio
    optimized_portfolio = optimize_portfolio(returns)
