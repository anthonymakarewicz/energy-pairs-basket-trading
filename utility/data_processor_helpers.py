import pandas as pd
import numpy as np
import yfinance as yf


def fetch_data(tickers, start=None, end=None, freq="1d", how="outer", **kwargs):
    """
    Fetches closing prices using yahoo finance for the given list of tickers.

    Args:
        tickers (list): List of ticker symbols.
        start (str, optional): Start date in 'YYYY-MM-DD' format.
        end (str, optional): End date in 'YYYY-MM-DD' format.
        freq (str, optional): Data frequency ('1d', '1wk', '1mo'). Default is '1d'.

    Returns:
        pd.DataFrame: DataFrame containing closing prices with ticker names as columns.
    """
    prices = None
    for ticker in tickers:
        try:
            closing_price = yf.download(ticker, start=start, end=end, interval=freq, **kwargs)["Close"]
            if prices is None:
                prices = closing_price
            elif not closing_price.empty:
                prices = pd.merge(prices, closing_price, left_index=True, right_index=True, how=how, **kwargs)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

    prices.index = prices.index.tz_localize(None)

    """

    NEED TO PASS THREADS ARG, REMOVE HOW 
    prices = yf.download(tickers=tickers, start=start, end=end, interval=freq, threads=True, **kwargs)["Close"]
    prices = prices.dropna(axis=1, how="all")
    valid_tickers = [ticker for ticker in tickers if ticker in prices.columns]
    prices = prices[valid_tickers]
    prices.index = prices.index.tz_localize(None)
    """
    return prices


def compute_log_returns(price):
    return np.log(price / price.shift(1)).fillna(0)


def compute_spread(prices, tickers, hedge_ratios):
    """
    Compute the spread based on the provided hedge ratios.

    Args:
        prices (pd.DataFrame): DataFrame containing asset prices.
        tickers (list): List of tickers to compute the spread for.
        hedge_ratios (list): List of hedge ratios corresponding to the assets.

    Returns:
        pd.DataFrame: The computed spread.
    """
    if len(tickers) != len(hedge_ratios):
        raise ValueError("The number of hedge ratios must match the number of tickers.")

    selected_prices = prices[tickers]
    spread = selected_prices.dot(hedge_ratios)
    spread.name = "spread"

    return spread.to_frame()


def compute_spreads(prices, pairs):
    from utility.statistics_helpers import get_beta_ols

    spreads = None
    for pair in pairs:
        pair = list(pair)
        prices_pair = prices[pair].interpolate().dropna()

        beta = get_beta_ols(prices_pair, pair)
        hedge_ratios = [1, -beta]
        spread = compute_spread(prices_pair, pair, hedge_ratios=hedge_ratios)
        spread.columns = [f"{pair[0]} vs {pair[1]}"]
        
        if spreads is None:
            spreads = spread
        else:
            spreads = pd.merge(spreads, spread, left_index=True, right_index=True, how="outer")

    return spreads