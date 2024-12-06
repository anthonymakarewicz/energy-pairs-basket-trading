import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from plotly.subplots import make_subplots
from utility.data_processor_helpers import compute_log_returns, compute_spread
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_normalized_prices(prices, tickers=None, plot_returns=False):
    """
    Plot normalized prices of assets using Plotly, with an option to add a subplot for log returns.

    Args:
        prices (pd.DataFrame): DataFrame containing price data.
        tickers (list, optional): List of tickers to plot. If None, all columns in `prices` are plotted.
        plot_returns (bool, optional): If True, adds a subplot for the log returns. Default is False.
    """
    if tickers is None:
        tickers = prices.columns

    if plot_returns:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Normalized Prices", "Log Returns"),
            vertical_spacing=0.15
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=("Normalized Prices",)
        )

    for ticker in tickers:
        normalized_price = prices[ticker] / prices[ticker].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=normalized_price.index,
                y=normalized_price,
                mode="lines",
                name=f"{ticker}",
                legendgroup="Normalized",  
                showlegend=True  
            ),
            row=1, col=1
        )

    if plot_returns:
        for ticker in tickers:
            log_returns = compute_log_returns(prices[ticker])
            fig.add_trace(
                go.Scatter(
                    x=log_returns.index,
                    y=log_returns,
                    mode="lines",
                    name=f"{ticker}",
                    legendgroup="Log Returns",  
                    showlegend=True
                ),
                row=2, col=1
            )

    fig.update_layout(
        title="Normalized Price Comparison" if not plot_returns else "Comparison of Normalized Prices and Logarithmic Returns",
        height=600 if not plot_returns else 800,
        width=1200,
        legend_tracegroupgap=300
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    if plot_returns:
        fig.update_yaxes(title_text="Return", row=2, col=1)
    fig.update_xaxes(title_text="Date")

    fig.show()


def plot_spread(prices, tickers, hedge_ratios):
    """
    Plot the spread and the tickers involved in its computation.

    Args:
        prices (pd.DataFrame): DataFrame containing asset prices.
        tickers (list): List of tickers to compute the spread for.
        hedge_ratios (list): List of hedge ratios corresponding to the assets.
    """
    spread = compute_spread(prices, tickers, hedge_ratios)

    spread_equation = []
    for i, (beta, ticker) in enumerate(zip(hedge_ratios, tickers)):
        if i == 0 and beta > 0:
            spread_equation.append(f"{beta:.2f} * {ticker}")
        elif beta < 0:
            spread_equation.append(f"- {abs(beta):.2f} * {ticker}")
        else:
            spread_equation.append(f"+ {beta:.2f} * {ticker}")
    spread_equation = " ".join(spread_equation)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread["spread"],
            mode="lines",
            name="Spread",
            line=dict(color="blue")
        )
    )
    
    spread_mean = spread["spread"].mean()
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=[spread_mean] * len(spread),
            mode="lines",
            name="Mean",
            line=dict(color="red", dash="dash")
        )
    )

    fig.update_layout(
        title=f"Spread = {spread_equation}",
        xaxis_title="Date",
        yaxis_title="Spread",
        height=600,
        width=1200,
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top"
        )
    )

    fig.show()


def plot_spreads(spreads, num_cols=2, height=800, width=1200):
    """
    Plot multiple spreads using subplots for better visualization.

    Args:
        spreads (pd.DataFrame): DataFrame containing spreads for all pairs as columns.
        pairs (list of tuples): List of pairs corresponding to the spreads.
        num_cols (int): Number of columns in the subplot grid.
    """
    pairs = spreads.columns
    num_rows = -(-len(pairs) // num_cols)  # Ceiling division for rows

    fig = sp.make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=pairs, 
        shared_xaxes=False, 
        shared_yaxes=False
    )

    for idx, pair in enumerate(pairs):
        row = idx // num_cols + 1
        col = idx % num_cols + 1

        spread = spreads[pair].dropna()
        spread_mean = spread.mean()

        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread,
                mode="lines",
                name=f"Spread: {pair}",
                line=dict(color="blue")
            ),
            row=row,
            col=col
        )

        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=[spread_mean] * len(spread),
                mode="lines",
                name=f"Mean: {pair}",
                line=dict(color="red", dash="dash")
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title="Spreads",
        height=height,
        width=width,
        showlegend=False
    )

    fig.show()


def plot_rolling_correlation(prices, tickers=None, window=30, height=4, aspect=2):
    """
    Plot rolling correlations between asset pairs over time.

    Args:
        prices (pd.DataFrame): DataFrame of asset prices with a DateTime index.
        tickers (list): List of ticker names to include in the rolling correlation plots.
        window (int): Rolling window size for computing correlations.
    """
    if tickers is None:
        tickers = prices.columns

    rolling_corr = {}
    overall_corr = {}

    for i, asset1 in enumerate(tickers):
        for j, asset2 in enumerate(tickers):
            if i < j:  # Only calculate for unique pairs
                corr_series = prices[asset1].rolling(window).corr(prices[asset2])
                rolling_corr[f"{asset1} vs {asset2}"] = corr_series
                overall_corr[f"{asset1} vs {asset2}"] = prices[asset1].corr(prices[asset2])

    rolling_corr_df = pd.DataFrame(rolling_corr, index=prices.index)
    rolling_corr_df = rolling_corr_df.dropna()

    melted_corr = rolling_corr_df.reset_index().melt(
        id_vars="Date", var_name="Pair", value_name="Rolling Correlation"
    )

    num_pairs = len(rolling_corr_df.columns)
    col_wrap = 3 if num_pairs > 6 else num_pairs
    pairplot = sns.FacetGrid(
        melted_corr,
        col="Pair",
        col_wrap=col_wrap,
        height=height,
        aspect=aspect
    )

    pairplot.map_dataframe(
        sns.lineplot,
        x="Date",
        y="Rolling Correlation"
    )

    # Add long-run correlation lines
    for ax, pair in zip(pairplot.axes.flat, rolling_corr_df.columns):
        ax.axhline(
            y=overall_corr[pair],
            color="red",
            linewidth=2,
            label=f"Long Run Correlation = {overall_corr[pair]:.2f}"
        )
        ax.legend(loc="lower right", fontsize=8)

    pairplot.set_titles("{col_name}")
    pairplot.set_axis_labels("Date", "Rolling Correlation")
    pairplot.figure.suptitle(f"Rolling Correlations (window={window})", y=1.02)


def plot_correlograms(prices, tickers=None, pacf=True, lags=30):
    """
    Plot ACF and optionally PACF for multiple time series in a single row for each.

    Args:
        prices (pd.DataFrame): DataFrame containing price data.
        tickers (list, optional): List of tickers to plot. If None, all columns in `prices` are used.
        pacf (bool): If True, includes PACF plots in a separate row. Otherwise, plots only ACF.
        lags (int): Number of lags to include in the ACF/PACF plots.
    """
    if tickers is None:
        tickers = prices.columns

    nrows = 2 if pacf else 1
    ncols = len(tickers)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] 

    for idx, ticker in enumerate(tickers):
        plot_acf(prices[ticker], lags=lags, ax=axes[idx], title=f"ACF: {ticker}")

    if pacf:
        for idx, ticker in enumerate(tickers):
            plot_pacf(prices[ticker], lags=lags, ax=axes[idx + ncols], title=f"PACF: {ticker}")

    plt.tight_layout()
    plt.show()


def plot_heatmap(data_matrix, method_name, pairs=None, threshold=None, color_scale="Viridis", mask_lower=False):
    """
    Plot a heatmap with annotations highlighting pairs based on the threshold.

    Args:
        data_matrix (pd.DataFrame): Matrix of values (correlation, p-values, or Hurst exponents).
        method_name (str): Name of the method ("Correlation", "Cointegration", "Hurst").
        pairs (list): List of selected pairs that meet the threshold.
        threshold (float): Threshold value used for filtering pairs.
        color_scale (str): Colorscale for the heatmap.
    """
    valid_methods = {"correlation": (-1, 1),
                     "cointegration": (0, 1), 
                     "hurst": (0, 1)}
    
    method_name_normalized = method_name.strip().lower()
    if method_name_normalized not in valid_methods:
        raise ValueError(f"Invalid method_name: {method_name}. Must be one of {list(valid_methods.keys())}.")
    zmin, zmax = valid_methods[method_name_normalized] 

    if mask_lower and method_name_normalized != "correlation":
        raise ValueError("Masking the lwoer triangular matrix is only available with 'correlation' method")

    matrix = data_matrix.copy()
    non_diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
    matrix = matrix.where(non_diagonal_mask)

    if mask_lower:
        upper_tri_mask = np.triu(np.ones_like(matrix, dtype=bool))
        matrix = matrix.where(upper_tri_mask)

    annotations = []
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            if i == j or (mask_lower and i > j):
                continue            

            corr_val = data_matrix.iloc[i, j]
            color = "black"
            if pairs is not None:
                if (data_matrix.index[i], data_matrix.columns[j]) in pairs:
                    color = "limegreen"
                else:
                    color = "red"

            text_corr = ""
            corr_str = str(corr_val)
            if corr_str.startswith("-"):
                text_corr = f"-.{corr_str[3:5]}"
            else:
                text_corr = f".{corr_str[2:4]}"

            annotations.append(
                go.layout.Annotation(
                    x=data_matrix.columns[j],
                    y=data_matrix.index[i],
                    text=text_corr,
                    showarrow=False,
                    font=dict(size=10, color=color)
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale=color_scale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=method_name_normalized.capitalize())
        )
    )

    title = f"{method_name_normalized.capitalize()} Heatmap"
    if threshold is not None:
        title += f" (Threshold = {threshold:.2f})"

    fig.update_layout(
        annotations=annotations,
        plot_bgcolor="white",
        title=title,
        xaxis=dict(title="Assets", tickangle=-45),
        yaxis=dict(title="Assets", autorange="reversed"),
        width=900,
        height=800
    )
    
    fig.show()


def plot_hist(data_matrix, method_name, threshold=None, num_bins=50):
    """
    Plot a histogram of values (correlation, p-values, or Hurst exponents).

    Args:
        data_matrix (pd.DataFrame): Matrix of values (correlation, p-values, or Hurst exponents).
        method_name (str): Name of the method (e.g., "Correlation", "Cointegration").
        num_bins (int): Number of bins for the histogram.
    """
    valid_methods = {"correlation": "Correlation Coefficient",
                     "cointegration": "P-value", 
                     "hurst": "Hurst Exponent"}
    
    method_name = method_name.strip().lower()
    if method_name not in valid_methods:
        raise ValueError(f"Invalid method_name: {method_name}. Must be one of {list(valid_methods.keys())}.")
    
    mask = np.triu(np.ones_like(data_matrix, dtype=bool), k=1)
    values = data_matrix.to_numpy()[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=num_bins,
            marker=dict(color="blue", line=dict(color="black", width=1)),
            opacity=0.75,
            name="Data Distribution"
        )
    )

    if threshold is not None:
        hist_data = np.histogram(values, bins=num_bins)
        max_frequency = max(hist_data[0])
        fig.add_trace(
            go.Scatter(
                x=[threshold, threshold],
                y=[0, max_frequency],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name=f"Threshold = {threshold:.2f}"
            )
        )

    fig.update_layout(
        title=f"{method_name.capitalize()} Histogram",
        xaxis_title=valid_methods[method_name].capitalize(),
        yaxis_title="Frequency",
        bargap=0.2,
        showlegend=True,
        width=900,
        height=500
    )
    
    fig.show()


def plot_pairwise_relationships(prices, tickers=None):
    """
    Plot pairwise scatter plots with regression lines for all assets, suppressing diagonal histograms.

    Args:
        prices (pd.DataFrame): DataFrame of asset prices.
        tickers (list): List of ticker names corresponding to columns in prices.
    """
    selected_prices = prices.copy()
    if tickers is not None:
        selected_prices = selected_prices[tickers]

    pairplot = sns.pairplot(
        selected_prices,
        kind="reg",
        plot_kws={'scatter_kws': {'s': 10, "alpha":0.2}, 'line_kws': {'color': 'red'}},
        diag_kind="kde",
        diag_kws={"linewidth": 0, "fill": False},
        height=3,
        aspect=1.5,
    )

    pairplot.figure.suptitle("Pairwise Scatter Plots with Regression Lines", y=1.02)
    plt.show()


def plot_fixed_zscore(z_score, signals, entry_threshold=2, exit_threshold=0.5, height=7, width=10):
    """
    Plot the z-score of the spread and entry/exit signals.

    Args:
        spread (pd.Series): The spread between two assets.
        signals (pd.DataFrame): Signals generated by the strategy.
        z_score (pd.Series): Z-score values.
        entry_threshold (float): Z-score threshold to enter a trade.
        exit_threshold (float): Z-score threshold to exit a trade.
        rolling (bool): Whether rolling mean/std were used in z-score calculation.
        window (int): Rolling window size for rolling z-score calculation.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))

    # Plot z-score
    ax1.plot(z_score, label="Z-Score", color="blue", zorder=1)

    # Plot entry and exit thresholds
    ax1.axhline(entry_threshold, color="green", linestyle="--", label=f"±{entry_threshold}σ Entry", zorder=2)
    ax1.axhline(-entry_threshold, color="green", linestyle="--", zorder=2)
    ax1.axhline(exit_threshold, color="red", linestyle="--", label=f"±{exit_threshold}σ Exit", zorder=2)
    ax1.axhline(-exit_threshold, color="red", linestyle="--", zorder=2)

    # Plot entry/exit points
    long_entries = z_score[signals["long"]]
    short_entries = z_score[signals["short"]]
    exits = z_score[signals["exit"]]

    ax1.scatter(long_entries.index, long_entries, color="lime", label="Long Entry", marker="^", s=50, edgecolors="black", zorder=3)
    ax1.scatter(short_entries.index, short_entries, color="orange", label="Short Entry", marker="v", s=50, edgecolors="black", zorder=3)
    ax1.scatter(exits.index, exits, color="purple", label="Exit", marker="D", s=50, edgecolors="black", zorder=3)

    ax1.set_title("Z-Score Strategy")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Z-Score")
    ax1.legend()
    ax1.grid()

    ax2.plot(signals["position"], label="Position", color="purple", drawstyle="steps-post")
    ax2.set_title("Position")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Position")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()


def plot_rolling_zscore(spread, signals, entry_threshold=2, exit_threshold=0.5, window=20, height=5, width=12):
    """
    Plot the spread with Bollinger Bands and entry/exit signals.

    Args:
        spread (pd.Series): The spread between two assets.
        signals (pd.DataFrame): Signals generated by the strategy.
        entry_threshold (float): Multiplier for standard deviation to create Bollinger Bands.
        window (int): Rolling window size for Bollinger Bands calculation.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()

    long_entry_band = rolling_mean - entry_threshold * rolling_std
    short_entry_band = rolling_mean + entry_threshold * rolling_std
    long_exit_band = rolling_mean - exit_threshold * rolling_std
    short_exit_band = rolling_mean + exit_threshold * rolling_std

    plt.figure(figsize=(width, height))
    plt.plot(spread, label="Spread", color="blue", zorder=1)

    plt.plot(long_entry_band, label=f"Entry Band (±{entry_threshold}σ)", color="green", linestyle="--", zorder=2)
    plt.plot(short_entry_band, color="green", linestyle="--", zorder=2)

    plt.plot(long_exit_band, label=f"Exit Band (±{exit_threshold}σ)", color="red", linestyle="--", zorder=2)
    plt.plot(short_exit_band, color="red", linestyle="--", zorder=2)

    # Plot entry/exit points
    long_entries = spread[signals["long"]]
    short_entries = spread[signals["short"]]
    exits = spread[signals["exit"]]

    plt.scatter(long_entries.index, long_entries, color="lime", label="Long Entry", marker="^", s=50, edgecolors="black", zorder=3)
    plt.scatter(short_entries.index, short_entries, color="orange", label="Short Entry", marker="v", s=50, edgecolors="black", zorder=3)
    plt.scatter(exits.index, exits, color="purple", label="Exit", marker="D", s=50, edgecolors="black", zorder=3)

    plt.title(f"Bollinger Bands Strategy (Window={window})")
    plt.xlabel("Date")
    plt.ylabel("Spread Value")
    plt.legend()
    plt.grid()
    plt.show()


def plot_performance_surface():
    entry_thresholds = np.linspace(1, 3, 50)    # Entry thresholds
    exit_thresholds = np.linspace(0.1, 1, 50)  # Exit thresholds with a focus on 0.5

    # Simulated performance metric
    performance = np.exp(-(entry_thresholds[:, None] - 2)**2 / 0.5) * np.exp(-(exit_thresholds - 0.5)**2 / 0.2)

    entry_mesh, exit_mesh = np.meshgrid(entry_thresholds, exit_thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    heatmap = axes[0].imshow(performance.T, extent=[1, 3, 0.1, 1], origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Performance Heatmap")
    axes[0].set_xlabel("Entry Threshold")
    axes[0].set_ylabel("Exit Threshold")
    fig.colorbar(heatmap, ax=axes[0], label="Performance Metric")

    ax3d = fig.add_subplot(122, projection="3d")
    ax3d.plot_surface(entry_mesh, exit_mesh, performance.T, cmap="viridis", alpha=0.9)
    ax3d.set_title("Performance 3D Surface")
    ax3d.set_xlabel("Entry Threshold")
    ax3d.set_ylabel("Exit Threshold")
    ax3d.set_zlabel("Performance Metric")
    ax3d.view_init(elev=20, azim=130)

    plt.tight_layout()
    plt.show()


def plot_time_series_cv(total_length=1000, train_size=756, val_size=252, test_size=252, step_size=252, split_type="train_test",
                        save=False):
    """
    Plot a visual representation of time series cross-validation with multiple folds.

    Args:
        total_length (int): Total length of the time series.
        train_size (int): Number of points in the training window.
        val_size (int): Number of points in the validation window.
        test_size (int): Number of points in the testing window.
        step_size (int): Number of points the windows shift for each fold.
        split_type (str): Type of split to visualize ("train_test" or "train_val_test").
    """
    if split_type not in ["train_test", "train_val_test"]:
        raise ValueError("Invalid split_type. Choose 'train_test' or 'train_val_test'.")

    # Determine number of folds
    if split_type == "train_test":
        n_splits = (total_length - train_size - test_size) // step_size + 1
    elif split_type == "train_val_test":
        n_splits = (total_length - train_size - val_size - test_size) // step_size + 1

    # Convert lengths to years
    train_years = train_size / 252
    val_years = val_size / 252
    test_years = test_size / 252

    fig, ax = plt.subplots(figsize=(12, 6))

    for fold in range(n_splits):
        start_train = fold * step_size
        end_train = start_train + train_size

        # Train + Test split
        if split_type == "train_test":
            start_test = end_train
            end_test = start_test + test_size

            # Training window
            ax.broken_barh([(start_train, train_size)], (n_splits - fold - 0.4, 0.8),
                           facecolors='blue', edgecolors='black', label=f'Training ({train_years:.1f} years)' if fold == 0 else "")
            # Testing window
            ax.broken_barh([(start_test, test_size)], (n_splits - fold - 0.4, 0.8),
                           facecolors='green', edgecolors='black', label=f'Testing ({test_years:.1f} years)' if fold == 0 else "")

        # Train + Validation + Test split
        elif split_type == "train_val_test":
            start_val = end_train
            end_val = start_val + val_size
            start_test = end_val
            end_test = start_test + test_size

            # Training window
            ax.broken_barh([(start_train, train_size)], (n_splits - fold - 0.4, 0.8),
                           facecolors='blue', edgecolors='black', label=f'Training ({train_years:.1f} years)' if fold == 0 else "")
            # Validation window
            ax.broken_barh([(start_val, val_size)], (n_splits - fold - 0.4, 0.8),
                           facecolors='orange', edgecolors='black', label=f'Validation ({val_years:.1f} years)' if fold == 0 else "")
            # Testing window
            ax.broken_barh([(start_test, test_size)], (n_splits - fold - 0.4, 0.8),
                           facecolors='green', edgecolors='black', label=f'Testing ({test_years:.1f} years)' if fold == 0 else "")

    title = "Walk-Forward" + (" Testing (Train + Test)" if split_type == "train_test" else " Cross Validation (Train + Validation + Test)")

    ax.set_title(title)
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {fold+1}" for fold in range(n_splits)][::-1])
    ax.set_xlabel("Time Index")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.tight_layout()
    plt.show()

    if save:
        images_path = Path("images")
        file_name = images_path / ("walk_forward_testing" if split_type == "train_test" else "walk_forward_cv")
        file_name = file_name.with_suffix(".png")
        fig.savefig(file_name, dpi=300)