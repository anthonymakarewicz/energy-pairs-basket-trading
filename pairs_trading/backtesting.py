import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import ParameterGrid

from .utility import estimate_ou_parameters_ols


class Backtester:
    """
    A backtester for evaluating pairs and baskest trading performance for a specific trading strategy and hedge ratio model.
    This class supports both walk forward with and without cross validation.
    In the case of cross-validation, it selects the most promising trading parameters based on performance on a validation window.

    Parameters:
    - prices (pd.DataFrame): DataFrame containing historical prices of assets.
    - trading_strategy: An instance of the Strategy class.
    - hedge_ratio_model: An instance of the HedgeRatioModel class.
    - train_window (int, optional): Number of days in the training window. Default is 252*3 (3 years).
    - val_window (int, optional): Number of days in the validation window. Default is 252 (1 year).
    - test_window (int, optional): Number of days in the test window. Default is 252 (1 year).
    - param_grid (dict, optional): Parameter grid for hyperparameter tuning. Default is None.
    - pairs_validation_methods (dict, optional): Methods for validating asset pairs. Default includes 'adf', 'half_life', and 'corr'.
    - max_trade_duration_multiplier (float, optional): Multiplier of half-life to limit trade duration. Default is 1.
    - max_drawdown_limit (float, optional): Maximum allowable drawdown before exiting a trade. Default is 0.1.
    - transaction_cost_per_unit (float, optional): Transaction cost per unit traded. Default is 0.01.

    Methods:

    Public Methods:
    - get_params(): Returns the parameters of the backtester as a dictionary
    - set_params(**params): Modifies the current parameters given the new ones specified in params
    - run_backtest(): Runs the backtest over the provided price data using the specified strategy and hedge ratio model.
    - calculate_performance_metrics(returns=None): Calculates various performance metrics for the backtest.
    - save_results(file_path): Saves the backtest results to a file specified in file_path.
    
    Private Methods:
    - __compute_hedge_ratio(data): Computes the hedge ratio using the hedge ratio model.
    - __calculate_spread(data, hedge_ratio): Calculates the spread based on asset prices and hedge ratio.
    - __is_valid_pair(spread): Validates the asset pair based on specified methods (ADF test, half-life, correlation).
    - __compute_half_life(spread): Computes the half-life of mean reversion for a given spread.
    - __optimize_parameters(train_data, val_data, hedge_ratio): Optimizes trading strategy parameters using a validation set.
    - __calculate_positions(signals, spreads, half_life): Calculates positions based on signals, spreads, and half-life, including exit conditions.
    - __calculate_returns(positions, spreads): Calculates net returns accounting for transaction costs.
    - __evaluate_performance(returns): Computes the Sharpe ratio given for the returns necessary for the optimize_parameters method.
    """
    def __init__(self,
                 prices, 
                 trading_strategy, 
                 hedge_ratio_model, 
                 train_window=252*3, 
                 val_window=252, 
                 test_window=252, 
                 param_grid=None, 
                 pairs_validation_methods=None,
                 max_trade_duration_multiplier=0, 
                 max_drawdown_limit=0, 
                 transaction_cost_per_unit=0.01):
        
        self.assets = prices.columns
        self.prices = prices
        self.trading_strategy = trading_strategy
        self.hedge_ratio_model = hedge_ratio_model

        if pairs_validation_methods is None:
            pairs_validation_methods = {"adf": 0.05, "half_life": [1, 5], "corr": 0.7}

        self.pairs_validation_methods = pairs_validation_methods
        self.max_trade_duration_multiplier = max_trade_duration_multiplier
        self.max_drawdown_limit = -abs(max_drawdown_limit) if max_drawdown_limit else None
        self.param_grid = param_grid
        self.transaction_cost_per_unit = transaction_cost_per_unit

        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window

        if self.param_grid is None:
            self.val_window = 0

        if (self.param_grid is not None) and (len(self.param_grid) * 252 > self.train_window):
            self.train_window = len(self.param_grid) * 252
            print(f"Training window set to {self.train_window} days based on the number of parameters.")
        
        total_required_length = self.train_window + self.val_window + self.test_window
        if len(prices) < total_required_length:
            raise ValueError(
                f"Not enough data for backtesting. "
                f"Required: {total_required_length} data points, but only {len(prices)} available. "
                "Please provide more data or adjust the window sizes."
            )
        
    ###################
    # Private methods #
    ###################

    def __compute_hedge_ratio(self, data):
        return self.hedge_ratio_model.compute_hedge_ratio(data, self.assets)

    def __calculate_spread(self, data, hedge_ratio):
        hedge_ratio = hedge_ratio.loc[data.index.intersection(hedge_ratio.index)]
        data = data.loc[hedge_ratio.index]
        spread = (data * hedge_ratio.to_numpy()).sum(axis=1)
        return spread  

    def __is_valid_pair(self, spread):
        are_valids = [True, True, True]

        # Augmented Dickey-Fuller test for stationarity
        if "adf" in self.pairs_validation_methods:
            adf_test = adfuller(spread)
            adf_p_value = adf_test[1]
            adf_threshold = self.pairs_validation_methods["adf"]
            if not (adf_p_value < adf_threshold):
                are_valids[0] = False

        # Half-life condition
        if "half_life" in self.pairs_validation_methods:
            half_life = self.__compute_half_life(spread)
            min_half_life = self.pairs_validation_methods["half_life"][0]
            max_half_life = self.pairs_validation_methods["half_life"][1]
            if not (min_half_life <= half_life <= max_half_life):
                are_valids[1] = False

        # Correlation condition
        if "corr" in self.pairs_validation_methods:
            corr_matrix = self.prices[self.assets].corr()
            corr_coeff = np.min(corr_matrix)
            corr_threshold = self.pairs_validation_methods["corr"]
            if corr_coeff < corr_threshold:
                are_valids[2] = False
            
        return all(are_valids)

    def __compute_half_life(self, spread):
        theta, _, _ = estimate_ou_parameters_ols(spread)
        half_life = np.log(2) / theta
        return half_life

    def __optimize_parameters(self, train_data, val_data):
        best_performance = -np.inf
        best_params = None
        train_val_data = pd.concat([train_data, val_data])
        
        for params in ParameterGrid(self.param_grid):
            strategy_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('strategy__')}
            hedge_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('hedge__')}
            delta = params.get('delta', None)

            # For Kalman Filter strategy and/or hedge ratio
            strategy_delta = strategy_params.get('delta', None)
            hedge_delta = hedge_params.get('delta', None)
            if (strategy_delta is not None and hedge_delta is not None) and (strategy_delta != hedge_delta):
                continue  
            if delta is not None:
                strategy_params["delta"] = delta
                hedge_params["delta"] = delta

            try:
                self.trading_strategy.set_params(**strategy_params)
                self.hedge_ratio_model.set_params(**hedge_params)
            except Exception as E:
                #print(f"Exception occured for parameters {params}: {e}")
                continue

            if self.hedge_ratio_model.method == "dynamic":
                # Compute hedge ratios over combined train and val data
                hedge_ratios = self.__compute_hedge_ratio(train_val_data)
                train_hedge_ratio = hedge_ratios.loc[train_data.index]
                val_hedge_ratio = hedge_ratios.loc[val_data.index]
            else:
                # Compute hedge ratio over train data
                hedge_ratios = self.__compute_hedge_ratio(train_data)
                train_hedge_ratio = hedge_ratios
                val_hedge_ratio = hedge_ratios  # Use the same hedge ratio for validation

            train_spread = self.__calculate_spread(train_data, train_hedge_ratio)
            val_spread = self.__calculate_spread(val_data, val_hedge_ratio)

            half_life = self.__compute_half_life(train_spread)
            half_life_series = pd.Series(half_life, index=val_data.index)

            train_val_spread = pd.concat([train_spread, val_spread])
            train_val_signals = self.trading_strategy.generate_signals(train_val_spread)
            val_signals = train_val_signals.loc[val_data.index]

            val_positions = self.__calculate_positions(val_signals, val_spread, half_life_series)
            returns = self.__calculate_returns(val_positions, val_spread)

            performance = self.__evaluate_performance(returns)
            if performance > best_performance:
                best_performance = performance
                best_params = params.copy()

        return best_params

    def __evaluate_performance(self, returns):
        if returns.std() == 0:
            return -np.inf
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        return sharpe_ratio

    def __calculate_positions(self, signals, spreads, half_life):
        positions = pd.Series(0, index=signals.index)
        position = 0               # Current position: 1 for long, -1 for short, 0 for neutral
        trade_duration = 0        # Duration of the current trade
        entry_price = None       # Entry price of the current trade
        peak_return = 0         # Peak return since entering the trade
        cumulative_return = 0  # Cumulative return since entering the trade
        
        if self.max_drawdown_limit:
            signals['max_drawdown_exit'] = 0
        if self.max_trade_duration_multiplier:
            signals['half_life_exit'] = 0

        for i in range(len(signals)):
            date = signals.index[i]
            spread_return = spreads.diff().iloc[i] if i > 0 else 0

            # Exit the current position
            if signals['exit'].iloc[i]:
                positions.iloc[i] = 0
                position = 0
                trade_duration = 0
                entry_price = None
                peak_return = 0
                cumulative_return = 0

            # Enter a new position
            elif signals['long'].iloc[i] or signals['short'].iloc[i]:
                trade_duration = 1
                entry_price = spreads.iloc[i]
                peak_return = 0
                cumulative_return = 0

                if signals['long'].iloc[i]:
                    positions.iloc[i] = 1
                    position = 1
                else:
                    positions.iloc[i] = -1
                    position = -1
            else:
                # Stay in the trade
                if position != 0:
                    trade_duration += 1

                    # Check for max-drawdown based exit
                    if self.max_drawdown_limit:
                        cumulative_return += position * spread_return / abs(entry_price) if entry_price != 0 else 0
                        peak_return = max(peak_return, cumulative_return)
                        drawdown = cumulative_return - peak_return

                        if drawdown <= self.max_drawdown_limit:
                            positions.iloc[i] = 0  # Exit position 
                            position = 0
                            trade_duration = 0
                            entry_price = None
                            peak_return = 0
                            cumulative_return = 0
                            signals.at[date, 'max_drawdown_exit'] = 1
                            
                    # Check for half-life based exit
                    if self.max_trade_duration_multiplier:
                        if (len(half_life) > 0
                            and trade_duration > half_life.loc[positions.index[i]] * self.max_trade_duration_multiplier):
                            positions.iloc[i] = 0  # Exit position
                            position = 0
                            trade_duration = 0
                            entry_price = None
                            peak_return = 0
                            cumulative_return = 0
                            signals.at[date, 'half_life_exit'] = 1
                        # Maintain current position
                        else:
                            positions.iloc[i] = position  

                # Stay out of the trade
                else:
                    positions.iloc[i] = 0
                    
        return positions
    
    def __calculate_returns(self, positions, spreads):
        shifted_positions = positions.shift().fillna(0)
        spread_changes = spreads.diff().fillna(0)
        transaction_costs = shifted_positions.abs() * self.transaction_cost_per_unit

        raw_returns = shifted_positions * spread_changes
        net_returns = raw_returns - transaction_costs
        return net_returns
    
    ##################
    # Public methods #
    ##################

    def get_params(self):
        return {
            'train_window': self.train_window,
            'val_window': self.val_window,
            'test_window': self.test_window,
            'max_trade_duration_multiplier': self.max_trade_duration_multiplier,
            'max_drawdown_limit': self.max_drawdown_limit,
            'transaction_cost_per_unit': self.transaction_cost_per_unit,
            'param_grid': self.param_grid,
            'pairs_validation_methods': self.pairs_validation_methods,
        }
    
    def set_params(self, **params):
        self.train_window = params.get('train_window', self.train_window)
        self.val_window = params.get('val_window', self.val_window)
        self.test_window = params.get('test_window', self.test_window)
        self.max_trade_duration_multiplier = params.get('max_trade_duration_multiplier', self.max_trade_duration_multiplier)
        self.max_drawdown_limit = params.get('max_drawdown_limit', self.max_drawdown_limit)
        self.transaction_cost_per_unit = params.get('transaction_cost_per_unit', self.transaction_cost_per_unit)
        self.param_grid = params.get('param_grid', self.param_grid)
        self.pairs_validation_methods = params.get('pairs_validation_methods', self.pairs_validation_methods)
    

    def run_backtest(self):
        """
        Runs the backtest over the provided price data using the specified strategy and hedge ratio model.

        This method iteratively trains, validates, and tests the trading strategy over rolling windows of the dataset.
        It computes spreads, hedge ratios, signals, and caches the results for performance evaluation.
        """
        self.spreads = []  
        self.signal_history = []  
        self.hedge_ratio_history = []
        self.half_life_history = []
        previous_spread_end = None  # To adjust test spreads when concatenating different windows
        self.best_params = []
        self.valid_backtests = []

        for start in range(0, len(self.prices) - self.train_window - self.val_window - self.test_window + 1, self.test_window):
            train_data = self.prices.iloc[start:start + self.train_window][self.assets]
            val_data = self.prices.iloc[start + self.train_window:start + self.train_window + self.val_window][self.assets]
            test_data = self.prices.iloc[start + self.train_window + self.val_window:start + self.train_window + self.val_window + self.test_window][self.assets]
            train_val_data = pd.concat([train_data, val_data]) if len(val_data) > 0 else train_data

            # Step 1: Hyperparameter tuning (if applicable)
            if self.param_grid:
                best_params = self.__optimize_parameters(train_data, val_data)
                if best_params:
                    strategy_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('strategy__')}
                    hedge_params = {k.split('__')[1]: v for k, v in best_params.items() if k.startswith('hedge__')}

                    self.trading_strategy.set_params(**strategy_params)
                    self.hedge_ratio_model.set_params(**hedge_params)
                    self.best_params.append(pd.DataFrame([best_params], index=test_data.index))
            else:
                self.trading_strategy.reset_params()
                self.hedge_ratio_model.reset_params()

            # Step 2: Compute hedge ratios and spreads
            if self.hedge_ratio_model.method == "dynamic":
                # Compute hedge ratios over combined data
                train_val_test_data = pd.concat([train_data, val_data, test_data])
                hedge_ratios = self.__compute_hedge_ratio(train_val_test_data)

                train_hedge_ratio = hedge_ratios.loc[train_data.index]
                val_hedge_ratio = hedge_ratios.loc[val_data.index]
                test_hedge_ratio = hedge_ratios.loc[test_data.index]

                # Set the new train spread as the combined one
                train_spread = self.__calculate_spread(train_data, train_hedge_ratio)
                val_spread = self.__calculate_spread(val_data, val_hedge_ratio)
                train_spread = pd.concat([train_spread, val_spread])

                test_spread = self.__calculate_spread(test_data, test_hedge_ratio)
                self.hedge_ratio_history.append(test_hedge_ratio)
            else:
                train_hedge_ratio = self.__compute_hedge_ratio(train_val_data)
                train_spread = self.__calculate_spread(train_val_data, train_hedge_ratio)
                test_spread = self.__calculate_spread(test_data, train_hedge_ratio)
                self.hedge_ratio_history.append(train_hedge_ratio)

            # Step 3: Adjust the spread for continuity
            if previous_spread_end is not None:
                adjustment = previous_spread_end - test_spread.iloc[0]
                test_spread += adjustment
            previous_spread_end = test_spread.iloc[-1]
            self.spreads.append(test_spread)

            # Step 4: Check if the pair is valid given the condition(s) 
            is_valid = self.__is_valid_pair(train_spread)
            if not is_valid:
                empty_signals = pd.DataFrame(0, index=test_data.index, columns=['long', 'short', 'exit'])
                self.signal_history.append(empty_signals)
                self.valid_backtests.append(False)
                continue
            self.valid_backtests.append(True)

            # Step 5: Compute the half life on the training spread
            if self.max_trade_duration_multiplier:
                half_life = self.__compute_half_life(train_spread)
                half_life_series = pd.Series(half_life, index=test_spread.index) 
                self.half_life_history.append(half_life_series)

            # Step 6: Generate the trading signals on the test set using trading strategy
            # Use the combined train and test spreads for signal generation
            train_test_spread = pd.concat([train_spread, test_spread])
            train_test_signals = self.trading_strategy.generate_signals(train_test_spread)
            test_signals = train_test_signals.loc[test_spread.index]
            self.signal_history.append(test_signals)

        # Step 7: Cache the results 
        self.spreads = pd.concat(self.spreads)
        epsilon = 1e-8
        rolling_mean = self.spreads.expanding(min_periods=1).mean()
        rolling_std = self.spreads.expanding(min_periods=1).std() + epsilon
        self.spreads = (self.spreads - rolling_mean) / rolling_std

        self.hedge_ratio_history = pd.concat(self.hedge_ratio_history).reindex(self.spreads.index).ffill()
        self.signal_history = pd.concat(self.signal_history)

        if len(self.half_life_history) > 0: 
            self.half_life_history = pd.concat(self.half_life_history)
            self.half_life_history = self.half_life_history.reindex(self.signal_history.index, fill_value=0)

        if self.best_params:
            self.best_params = pd.concat(self.best_params)

        self.positions = self.__calculate_positions(self.signal_history, self.spreads, self.half_life_history)
        self.returns = self.__calculate_returns(self.positions, self.spreads)

        self.trading_strategy.reset_params()
        self.hedge_ratio_model.reset_params()

    def calculate_performance_metrics(self, returns=None):
        """
        Calculates various performance metrics for the backtest.

        Parameters:
        - returns (pd.Series, optional): Series of returns. If None, uses the returns from the backtest.

        Returns:
        - pd.DataFrame: DataFrame containing performance metrics such as Sharpe Ratio, Sortino Ratio, Max Drawdown, etc.
        """
        if not hasattr(self, 'returns'):
            raise ValueError("Performance metrics not available. Please call the run_backtest() method.")        
        if returns is None:
            returns = self.returns

        metrics = pd.DataFrame(index=[0])

        # 1. Sharpe Ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan
        metrics["Sharpe Ratio"] = sharpe_ratio

        # 2. Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if downside_returns.std() != 0 else np.nan
        metrics["Sortino Ratio"] = sortino_ratio

        # 3. Drawdown based performance measures
        cumulative = (1 + returns.cumsum())
        rolling_max = cumulative.cummax()
        self.drawdowns = cumulative / rolling_max - 1

        # Max Drawdown
        max_drawdown = (-self.drawdowns).max()

        #Calmar Ratio
        calmar_ratio = returns.mean() / max_drawdown if max_drawdown != 0 else np.nan

        # Max Drawdown Duration
        in_drawdown = (cumulative < cumulative.cummax())
        self.drawdown_durations = in_drawdown.groupby((~in_drawdown).cumsum()).cumsum()
        max_drawdown_duration = self.drawdown_durations.max()

        # Ulcer Index
        squared_drawdowns = np.square(self.drawdowns)
        mean_squared_drawdowns = squared_drawdowns.mean()
        ulcer_index = np.sqrt(mean_squared_drawdowns)
        
        metrics["Calmar Ratio"] = calmar_ratio
        metrics["Max Drawdown"] = max_drawdown
        metrics["Max Drawdown Duration"] = max_drawdown_duration
        metrics["Ulcer Index"] = ulcer_index

        # 5. Average Win / Loss
        average_win = returns[returns > 0].mean()
        average_loss = returns[returns < 0].mean()
        metrics["Average Win"] = average_win
        metrics["Average Loss"] = average_loss

        # 6. Profit Factor (Total profit / Total loss)
        total_profit = returns[returns > 0].sum()
        total_loss = -returns[returns < 0].sum()  # Negative to make total loss positive
        profit_factor = total_profit / total_loss if total_loss != 0 else np.nan
        metrics["Profit Factor"] = profit_factor
        
        return metrics
    
    def save_results(self, filepath):
        """
        Saves the backtest results to a file.

        Parameters:
        - filepath (str): The path to the file where results will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


class PerformanceAnalyzer:
    """
    Analyzes and compares performance metrics of one or more backtests.

    Parameters:
    - backtests (list of tuples or dict): A collection of backtester instances with associated names.
    """
    def __init__(self, backtests):
        if isinstance(backtests, list):
            backtests = dict(backtests)
        self.backtests = backtests
        self.new_backtests = False
        self.metrics = None

    def set_backtests(self, backtests):
        if isinstance(backtests, list):
            backtests = dict(backtests)
        self.new_backtests = True
        self.backtests = backtests

    def add_backtests(self, backtests):
        if isinstance(backtests, list):
            backtests = dict(backtests)
        self.new_backtests = True
        self.backtests.update(backtests)

    def __compute_metrics(self):
        metrics_backtests = pd.DataFrame()

        for name, backtester in self.backtests.items():
            if not hasattr(backtester, 'returns'):
                raise ValueError(f"Backtester '{name}' has not run backtest yet.")
            metrics_backtest = backtester.calculate_performance_metrics()
            metrics_backtests = pd.concat([metrics_backtests, metrics_backtest])
        
        metrics_backtests.index = self.backtests
        self.metrics = metrics_backtests

    def get_metrics(self):
        """
        Computes and returns the performance metrics for all backtests.

        Returns:
        - pd.DataFrame: DataFrame containing performance metrics for each backtest.
        """
        if self.metrics is None or self.new_backtests:
            self.__compute_metrics()
            self.new_backtests = False
        return self.metrics

    def plot_results(self, strategy_names=None, window=100, height=10, width=7):
        """
        Plots equity curves, drawdowns, and rolling Sharpe ratios for specified backtests.

        Parameters:
        - window (int): Window size in days for calculating rolling Sharpe ratio.
        - height (int): Height of the plot.
        - width (int): Width of the plot.
        - strategy_names (list, optional): List of strategy names to plot. If None, plots all strategies.
        """
        if strategy_names is not None:
            if isinstance(strategy_names, (list, tuple)):
                invalid_strategies = [name for name in strategy_names if name not in self.backtests]
                if invalid_strategies:
                    raise ValueError(f"Invalid strategy names: {invalid_strategies}")
            elif isinstance(strategy_names, str):
                if strategy_names not in self.backtests:
                    raise ValueError(f"Invalid strategy name: {strategy_names}")
                strategy_names = [strategy_names]  # Convert to list for consistency
            else:
                raise TypeError("strategy_names must be a string, list, or tuple")
        else:
            strategy_names = list(self.backtests.keys())  # Default to all strategies
            
        backtests_to_plot = {name: self.backtests[name] for name in strategy_names if hasattr(self.backtests[name], "returns")}

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(height, width))

        # Plot Equity Curves
        for name, backtester in backtests_to_plot.items():
            cumulative_returns = backtester.returns.cumsum() + 1
            label = name if len(backtests_to_plot) > 1 else "Equity Curve"
            ax1.plot(cumulative_returns, label=label)
        ax1.set_title("Equity Curves: P&L for 1 unit of spread")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Returns")
        ax1.legend()
        ax1.grid(True)

        # Plot Drawdowns
        for name, backtester in backtests_to_plot.items():
            cumulative_returns = backtester.returns.cumsum() + 1
            rolling_max = cumulative_returns.cummax()
            drawdowns = cumulative_returns / rolling_max - 1
            label = None if len(backtests_to_plot) > 1 else "Drawdown"
            color = "purple" if len(backtests_to_plot) == 1 else None
            ax2.plot(drawdowns, label=label, color=color)
        ax2.set_title("UnderWater Curves")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown")
        if len(backtests_to_plot) == 1:
            ax2.legend()
        ax2.grid(True)

        # Plot Rolling Sharpe Ratios
        for name, backtester in backtests_to_plot.items():
            returns = backtester.returns
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
            label = None if len(backtests_to_plot) > 1 else "Rolling Sharpe Ratio"
            color = "red" if len(backtests_to_plot) == 1 else None
            ax3.plot(rolling_sharpe, label=label, color=color)
        ax3.set_title(f"Rolling Sharpe Ratios (window={window})")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Sharpe Ratio")
        if len(backtests_to_plot) == 1:
            ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_signals(self, strategy_name=None, start=None, end=None, height=6, width=12):
        """
        Plots the spread with entry and exit signals for one or multiple backtests.

        Parameters:
        - strategy_name (str or list, optional): Name(s) of the strategy/backtest(s) to plot. If None, plots all.
        - start (optional): Start date for plotting.
        - end (optional): End date for plotting.
        """
        strategies = []
        if strategy_name is None:
            strategies = list(self.backtests.keys())
        elif isinstance(strategy_name, str):
            strategies = [strategy_name]
        elif isinstance(strategy_name, list):
            strategies = strategy_name

        invalid_strategies = [s for s in strategies if s not in self.backtests]
        if invalid_strategies:
            raise ValueError(f"Strategy/strategies not found: {invalid_strategies}")

        n_strategies = len(strategies)
        n_rows = math.ceil(math.sqrt(n_strategies))
        n_cols = math.ceil(n_strategies / n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)
        axes = axes.flatten()

        for i, strategy in enumerate(strategies):
            backtester = self.backtests[strategy]

            if start is not None and end is not None:
                spreads = backtester.spreads[start:end]
                signals = backtester.signal_history[start:end]
            elif start is not None:
                spreads = backtester.spreads[start:]
                signals = backtester.signal_history[start:]
            elif end is not None:
                spreads = backtester.spreads[:end]
                signals = backtester.signal_history[:end]
            else:
                spreads = backtester.spreads
                signals = backtester.signal_history

            long_entries = (signals['long'] == 1)
            short_entries = (signals['short'] == 1)
            exits = (signals['exit'] == 1)

            max_drawdown_exit = None
            half_life_exit = None
            actual_exits = None

            if 'max_drawdown_exit' in signals.columns:
                max_drawdown_exit = (signals['max_drawdown_exit'] == 1)
            if 'half_life_exit' in signals.columns:
                half_life_exit = (signals['half_life_exit'] == 1)

            if max_drawdown_exit is not None and half_life_exit is not None:
                actual_exits = max_drawdown_exit | half_life_exit
            elif max_drawdown_exit is not None:
                actual_exits = max_drawdown_exit
            elif half_life_exit is not None:
                actual_exits = half_life_exit

            ax = axes[i]
            ax.plot(spreads, label='Spread')
            ax.plot(spreads[long_entries], 'g^', markersize=5, label='Long Entry')
            ax.plot(spreads[short_entries], 'rv', markersize=5, label='Short Entry')
            ax.plot(spreads[exits], 'bo', markersize=5, label=('Theoretical' if actual_exits is not None else '') + ' Exit')
            if actual_exits is not None:
                ax.plot(spreads[actual_exits], 'mo', markersize=5, label='Actual Exit')
            ax.set_title(f"Spread: {strategy}")
            ax.legend()

        for j in range(len(strategies), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_params(self, strategy_name, height=7, width=10):
        """
        Plots the evolution of the best parameters over time for a given strategy.

        Args:
            strategy_name (str): The name of the strategy to visualize the best parameters.
        """
        if strategy_name not in self.backtests.keys():
            raise ValueError(f"{strategy_name} is not present")

        backtester = self.backtests[strategy_name]
        best_params = backtester.best_params
        param_columns = best_params.columns
        
        years = np.unique(best_params.index.year)
        tick_positions = [pd.Timestamp(f"{year}-01-01") for year in years]
        
        n_params = len(param_columns)
        n_cols = 2
        n_rows = (n_params + n_cols - 1) // n_cols 
        fig = plt.figure(figsize=(width, height))
        gs = GridSpec(n_rows, n_cols, figure=fig)

        for idx, param in enumerate(param_columns):
            row, col = divmod(idx, n_cols)
            ax = fig.add_subplot(gs[row, col])
            ax.plot(best_params.index, best_params[param], label=param)
            ax.set_title(f"Optimal {param}")
            ax.set_ylabel("Parameter Value")
            ax.set_xticks(tick_positions)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()