import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def adf_test(prices, tickers=None):
    """
    Perform the Augmented Dickey-Fuller (ADF) test on selected time series.

    Args:
        prices (pd.DataFrame): DataFrame containing price data.
        tickers (list, optional): List of tickers to test. If None, all columns in `prices` are used.

    Returns:
        dict: A dictionary with ADF test results for each ticker.
    """
    if tickers is None:
        tickers = prices.columns

    results = {}
    for ticker in tickers:
        data = prices[ticker].interpolate().dropna()
        adf_result = adfuller(data)
        results[ticker] = {
            'Test Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical Values': adf_result[4]
        }

        print(f"ADF Test Results for {ticker}:")
        print(f"Test Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print("Critical Values:")
        for key, value in adf_result[4].items():
            print(f"    {key}: {value:.4f}")
        print()

    return results


def johansen_test(prices, tickers, det_order=0, k_ar_diff=1, significance_level=0.05):
    """
    Perform the Johansen test for cointegration and display results.

    Args:
        prices (pd.DataFrame): DataFrame containing the price data of assets.
        tickers (list): List of asset tickers to include in the test.
        det_order (int, optional): Deterministic trend assumption (default=0).
        k_ar_diff (int, optional): Number of lagged differences in the VECM (default=1).
        significance_level (float, optional): Significance level for critical value comparison (default=0.05).

    Returns:
        pd.DataFrame: A summary table of test statistics and critical value comparisons.
        np.array: The resulting cointegration coefficents
    """
    if not set(tickers).issubset(prices.columns):
        raise ValueError("Some tickers are not found in the prices DataFrame.")

    selected_prices = prices[tickers]

    result = coint_johansen(selected_prices, det_order=det_order, k_ar_diff=k_ar_diff)

    significance_index = {0.01: 0, 0.05: 1, 0.10: 2}[significance_level]

    trace_df = pd.DataFrame({
        "Hypothesis (r)": [f"r ≤ {i}" for i in range(len(result.lr1))],
        "Trace Statistic": result.lr1,
        "Critical Value (5%)": result.cvt[:, significance_index],
        "Reject H0?": result.lr1 > result.cvt[:, significance_index]
    })
    max_eigen_df = pd.DataFrame({
        "Hypothesis (r)": [f"r = {i}" for i in range(len(result.lr2))],
        "Max Eigenvalue Statistic": result.lr2,
        "Critical Value (5%)": result.cvm[:, significance_index],
        "Reject H0?": result.lr2 > result.cvm[:, significance_index]
    })
    beta_vec = result.evec[:, 0]

    return trace_df, max_eigen_df, beta_vec


def ljung_box_test(prices, tickers=None, lags=20, plot=True):
    """
    Perform the Ljung-Box test on selected time series and optionally visualize p-values in a grid.

    Args:
        prices (pd.DataFrame): DataFrame containing price data.
        tickers (list, optional): List of tickers to test. If None, all columns in `prices` are used.
        lags (int): Maximum number of lags for the Ljung-Box test.
        plot (bool): Whether to plot p-values for different lags.

    Returns:
        dict: A dictionary with Ljung-Box test results for each ticker.
    """
    if tickers is None:
        tickers = prices.columns

    results = {}
    
    num_tickers = len(tickers)
    grid_cols = 2  
    grid_rows = (num_tickers + grid_cols - 1) // grid_cols  

    if plot:
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 6 * grid_rows))
        axes = axes.flatten()

    for idx, ticker in enumerate(tickers):
        data = prices[ticker]
        ljung_box_result = sm.stats.acorr_ljungbox(data, lags=lags, return_df=True)
        results[ticker] = ljung_box_result

        if not plot:
            print(f"Ljung-Box Test Results for {ticker}:")
            print(ljung_box_result)
            print()

        if plot:
            ax = axes[idx]
            ax.plot(ljung_box_result.index, ljung_box_result['lb_pvalue'], marker='o', label='p-value')
            ax.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold (0.05)')
            ax.set_title(f'Ljung-Box Test: {ticker}')
            ax.set_xlabel('Lag')
            ax.set_ylabel('p-value')
            ax.set_xticks(ljung_box_result.index)  
            ax.legend()
            ax.grid()

    if plot:
        for ax in axes[num_tickers:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    return results


def get_beta_ols(prices, tickers):
    """
    Perform OLS regression and extract the beta coefficient.

    Args:
        prices (pd.DataFrame): DataFrame containing price data for the assets.
        tickers (list): List of two tickers [asset_1, asset_2] for which the hedge ratio is computed.

    Returns:
        float: Optimized hedge ratio (beta coefficient).
        pd.Series: Residuals of the regression.
    """
    if len(tickers) != 2:
        raise ValueError("Engle-Granger method requires exactly two assets.")
    
    reduced_prices = prices[tickers].interpolate().dropna()
    y = reduced_prices[tickers[0]]  # Dependent variable
    x = reduced_prices[tickers[1]]  # Independent variable
    x = sm.add_constant(x) 

    model = sm.OLS(y, x).fit()
    beta = model.params[tickers[1]]

    return beta


def compute_corr_matrix(prices, threshold=None):
    """
    Compute the correlation matrix and filter pairs based on a threshold.

    Args:
        prices (pd.DataFrame): DataFrame containing asset prices.
        threshold (float, optional): Minimum correlation threshold for selecting pairs.

    Returns:
        tuple: (correlation_matrix, selected_pairs)
            - correlation_matrix: DataFrame of correlation values.
            - selected_pairs: List of pairs that meet the threshold.
    """
    assets = prices.columns
    corr_matrix = prices.corr()
    selected_pairs = set()

    if threshold is not None:
        for i, asset_1 in enumerate(assets):
            for j, asset_2 in enumerate(assets):
                if i != j:  # Compute only for upper triangle
                    if threshold is not None and corr_matrix.iloc[i, j] > threshold:
                        selected_pairs.add((corr_matrix.index[i], corr_matrix.columns[j]))

    return corr_matrix, selected_pairs


def compute_coint_matrix(prices, threshold=None):
    """
    Compute the cointegration p-value matrix for a set of assets.

    Args:
        prices (pd.DataFrame): DataFrame where each column represents a time series.

    Returns:
        pd.DataFrame: Matrix of cointegration p-values.
    """
    selected_pairs = []
    assets = prices.columns
    coint_matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)

    for i in range(len(assets)):
        for j in range(len(assets)):
            if i != j:
                try:
                    asset1 = assets[i]
                    asset2 = assets[j]
                    merged = pd.concat([prices[asset1], prices[asset2]], axis=1, join="inner").dropna()

                    _, pvalue, _ = coint(merged[asset1], merged[asset2]) 
                    coint_matrix.loc[asset1, asset2] = pvalue
                    
                    if threshold is not None and pvalue <= threshold:
                        selected_pairs.append((asset1, asset2))

                except Exception as e:
                    print(f"Error computing cointegration for {asset1} and {asset2}: {e}")
                    coint_matrix.loc[asset1, asset2] = np.nan

    return coint_matrix, selected_pairs


def compute_hurst(series, max_lag=30):
    """
    Compute the Hurst exponent for a given time series using the standard deviation or rescaled range method.

    Args:
        series (pd.Series): Time series data.
        max_lag (int): Maximum lag to consider.
        method (str): Method to use for computation ("std" for standard deviation, "rs" for rescaled range).

    Returns:
        float: Estimated Hurst exponent.
    """
    lags = range(2, max_lag + 1)
    tau = [np.sqrt(series.diff(lag).dropna().to_numpy().std()) for lag in lags]
    hurst_exp = np.polyfit(np.log(lags), np.log(tau), 1)[0]    
    return hurst_exp * 2


def compute_hurst_matrix(prices, threshold=None, max_lag=30):
    """
    Compute a matrix of Hurst exponents for all assets and filter pairs based on a threshold.

    Args:
        prices (pd.DataFrame): DataFrame of asset prices with columns as assets.
        threshold (float, optional): Threshold for selecting pairs.

    Returns:
        pd.DataFrame: Matrix of Hurst exponents.
        list: List of pairs meeting the threshold criterion (if threshold is provided).
    """
    from utility.data_processor_helpers import compute_spread
    assets = prices.columns
    hurst_matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)
    valid_pairs = []

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i != j:  
                tickers = [asset1, asset2]
                merged = pd.concat([prices[asset1], prices[asset2]], axis=1, join="inner").dropna()
                
                if merged.shape[0] < max_lag + 1:  # Ensure enough data for Hurst computation
                    continue

                beta = get_beta_ols(merged, [asset1, asset2])
                hedge_ratios = [1, -beta]

                spread = compute_spread(merged, tickers=tickers, hedge_ratios=hedge_ratios)
                spread = spread.dropna()

                hurst_exp = compute_hurst(spread, max_lag=max_lag)
                hurst_matrix.loc[asset1, asset2] = hurst_exp
                #hurst_matrix.loc[asset2, asset1] = np.nan  # Lower triangle remains NaN
                
                if threshold is not None and hurst_exp < threshold:
                    valid_pairs.append((asset1, asset2))

    return hurst_matrix, valid_pairs if threshold is not None else hurst_matrix