import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from hmmlearn.hmm  import GaussianHMM

from .utility import estimate_ou_parameters_mle, estimate_ou_parameters_ols


class Strategy(ABC):
    @abstractmethod
    def get_params(self):
        """Return strategy parameters."""
        pass

    @abstractmethod
    def set_params(self, **params):
        """Set strategy parameters."""
        pass

    @abstractmethod
    def reset_params(self):
        """Reset strategy parameters."""
        pass

    @abstractmethod
    def generate_signals(self, spread):
        """Generate entry and exit signals based on the spread."""
        pass


########################
# Technical Indicators #
########################

# --- 


class RelativeStrengthIndexStrategy(Strategy):
    def __init__(self, window=14, lower_threshold=30, upper_threshold=70, diff_exit_threshold=10):
        self.params = {
            'window': window,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'diff_exit_threshold': diff_exit_threshold
        }
        self.initial_params = self.params.copy()
        self.set_params(**self.params)

    def get_params(self):
        return self.params.copy()

    def set_params(self, **params):
        for param in ['window', 'lower_threshold', 'upper_threshold', 'diff_exit_threshold']:
            if param in params:
                self.params[param] = params[param]
                setattr(self, param, params[param])

        # Validation
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("The lower threshold must be less than the upper threshold.")
        if self.window <= 0:
            raise ValueError("Window must be a positive integer.")
        if not isinstance(self.diff_exit_threshold, (int, float)):
            raise ValueError("diff_exit_threshold must be a number.")

    def reset_params(self):
        self.set_params(**self.initial_params)

    def generate_signals(self, spread):
        spread_return = spread.diff()
        up = spread_return.clip(lower=0)         # Store wins
        down = -1 * spread_return.clip(upper=0) # Store losses

        # Calculate moving averages of gains and losses
        roll_up = up.rolling(window=self.window, min_periods=1).mean()
        roll_down = down.rolling(window=self.window, min_periods=1).mean()

        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        signals = pd.DataFrame(0, index=spread.index, columns=['long', 'short', 'exit'])
        position = 0

        for i in range(len(rsi)):
            if position == 0:
                # Enter long or short
                if rsi.iloc[i] < self.lower_threshold:
                    signals.at[signals.index[i], 'long'] = 1
                    position = 1
                elif rsi.iloc[i] > self.upper_threshold:
                    signals.at[signals.index[i],'short'] = 1
                    position = -1

            elif position == 1:
                # Exit long position
                if rsi.iloc[i] > 50 - self.diff_exit_threshold:
                    signals.at[signals.index[i],'exit'] = 1
                    position = 0

            elif position == -1:
                # Exit short position
                if rsi.iloc[i] < 50 + self.diff_exit_threshold:
                    signals.at[signals.index[i],'exit'] = 1
                    position = 0

        return signals


class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, short_window=13, long_window=50):
        self.params = {
            'short_window': short_window,
            'long_window': long_window
        }
        self.initial_params = self.params.copy()
        self.set_params(**self.params)

    def get_params(self):
        return self.params.copy()

    def set_params(self, **params):
        for param in ['short_window', 'long_window']:
            if param in params:
                self.params[param] = params[param]
                setattr(self, param, params[param])

        # Validation
        if self.short_window >= self.long_window:
            raise ValueError("The short window must be less than the long window.")
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Window sizes must be positive integers.")

    def reset_params(self):
        self.set_params(**self.initial_params)

    def generate_signals(self, spread):
        short_ma = spread.rolling(window=self.short_window, min_periods=1).mean()
        long_ma = spread.rolling(window=self.long_window, min_periods=1).mean()
        signals = pd.DataFrame(0, index=spread.index, columns=['long', 'short', 'exit'])
        position = 0  # 1 for long, -1 for short, 0 for neutral

        for i in range(1, len(signals)):
            if position == 0:
                # Enter long or short
                if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i - 1] <= long_ma.iloc[i - 1]:
                    signals.at[signals.index[i], 'long'] = 1
                    position = 1
                elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i - 1] >= long_ma.iloc[i - 1]:
                    signals.at[signals.index[i],'short'] = 1
                    position = -1

            elif position == 1:
                # Exit long position
                if short_ma.iloc[i] < long_ma.iloc[i]:
                    signals.at[signals.index[i], 'exit'] = 1
                    position = 0

            elif position == -1:
                # Exit short position
                if short_ma.iloc[i] > long_ma.iloc[i]:
                    signals.at[signals.index[i], 'exit'] = 1
                    position = 0

        return signals
    

class BollingerBandsStrategy(Strategy):
    def __init__(self, window=13, entry_threshold=1.5, exit_threshold=0.5):
        self.params = {
            'window': window,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold
        }
        self.initial_params = self.params.copy()
        self.set_params(**self.params)

    def get_params(self):
        return self.params.copy()

    def set_params(self, **params):
        for param in ['window', 'entry_threshold', 'exit_threshold']:
            if param in params:
                self.params[param] = params[param]
                setattr(self, param, params[param])

        # Validation
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("The exit threshold must be less than the entry threshold.")
        if self.window <= 0:
            raise ValueError("Window must be a positive integer.")

    def reset_params(self):
        self.set_params(**self.initial_params)
        
    def generate_signals(self, spread):
        rolling_mean = spread.rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = spread.rolling(window=self.window, min_periods=self.window).std()
        z_score = (spread - rolling_mean) / rolling_std

        signals = pd.DataFrame(0, index=spread.index, columns=['long', 'short', 'exit'])        
        position = 0 # 1 for long, -1 for short, 0 for neutral
        
        for i in range(len(z_score)):
            if position == 0:
                # Enter long or short
                if z_score.iloc[i] < -self.entry_threshold:
                    signals.at[signals.index[i], 'long'] = 1
                    position = 1
                elif z_score.iloc[i] > self.entry_threshold:
                    signals.at[signals.index[i], 'short'] = 1
                    position = -1

            elif position == 1 and z_score.iloc[i] > -self.exit_threshold:
                # Exit long position
                signals.at[signals.index[i], 'exit'] = 1
                position = 0

            elif position == -1 and z_score.iloc[i] < self.exit_threshold:
                # Exit short position
                signals.at[signals.index[i], 'exit'] = 1
                position = 0

        return signals


class KalmanFilterStrategy(Strategy):
    def __init__(self, kalman_model, entry_threshold=1.5, exit_threshold=0.5):
        self.kalman_model = kalman_model
        self.params = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold
        }
        self.initial_params = self.params.copy()
        self.set_params(**self.params)

    def set_params(self, **params):
        if 'delta' in params:
            delta = params['delta']
            if delta != self.kalman_model.delta:
                self.kalman_model.set_params(delta=delta)

        # Handle only parameters relevant to this strategy
        for param in ['entry_threshold', 'exit_threshold']:
            if param in params:
                self.params[param] = params[param]
                setattr(self, param, params[param])

        if self.exit_threshold > self.entry_threshold:
            raise ValueError("The exit threshold must be less than the entry threshold.")

    def get_params(self):
        return self.params.copy()
    
    def reset_params(self):
        self.set_params(**self.initial_params)
        self.kalman_model.reset_params()

    def generate_signals(self, spread):
        # Here spread is not used, since the trading signals are generated from KalmanFilterModel
        prediction_errors = self.kalman_model.get_prediction_errors()
        prediction_std = np.sqrt(self.kalman_model.get_prediction_variances())

        z_scores = prediction_errors / prediction_std

        signals = pd.DataFrame(0, index=z_scores.index, columns=['long', 'short', 'exit'])
        position = 0
        
        for i in range(len(z_scores)):
            if position == 0:
                # Enter long position
                if z_scores.iloc[i] < -self.params["entry_threshold"]:
                    signals.at[z_scores.index[i], 'long'] = 1
                    position = 1
                # Enter short position
                elif z_scores.iloc[i] > self.params["entry_threshold"]:
                    signals.at[z_scores.index[i], 'short'] = 1
                    position = -1
            elif position == 1:
                # Exit long position
                if z_scores.iloc[i] >= -self.params["exit_threshold"]:
                    signals.at[z_scores.index[i], 'exit'] = 1
                    position = 0
            elif position == -1:
                # Exit short position
                if z_scores.iloc[i] <= self.params["exit_threshold"]:
                    signals.at[z_scores.index[i], 'exit'] = 1
                    position = 0

        return signals.fillna(0)


class HMMZScoreStrategy(Strategy):
    def __init__(self, 
                 n_states=2, 
                 window=100, 
                 state_prob_threshold=0.7, 
                 entry_threshold=2.0, 
                 exit_threshold=0.5, 
                 use_predicted_prob=False,
                 random_state=None):
        self.params = {
            'n_states': n_states,
            'window': window,
            'state_prob_threshold': state_prob_threshold,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'use_predicted_prob': use_predicted_prob,
            'random_state': random_state
        }
        self.initial_params = self.params.copy()
        self.set_params(**self.params)

    def get_params(self):
        return self.params.copy()

    def set_params(self, **params):
        avail_params = ['n_states', 'window','state_prob_threshold', 'entry_threshold',
                        'exit_threshold', 'use_predicted_prob', 'random_state']
        for param in avail_params:
            if param in params:
                self.params[param] = params[param]
                setattr(self, param, params[param])

        # Validation
        if not isinstance(self.n_states, int) or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer.")
        if self.window <= 0:
            raise ValueError("Window must be a positive integer.")
        if not (0 <= self.state_prob_threshold <= 1):
            raise ValueError("state_prob_threshold must be between 0 and 1.")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be less than entry_threshold.")
        if not isinstance(self.use_predicted_prob, bool):
            raise ValueError("use_predicted_prob must be a boolean.")

    def reset_params(self):
        self.set_params(**self.initial_params)
    
    def generate_signals(self, spread):
        if self.window >= len(spread):
            raise ValueError("Window size too small, please choose a lower one")

        signals = pd.DataFrame(0, index=spread.index, columns=['long', 'short', 'exit'])
        column_names = [f"State_{s}" for s in range(self.n_states)]
        position = 0  # 1 for long, -1 for short, 0 for neutral

        self.state_means = []
        self.state_vars = []
        self.state_probs = []

        for i in range(self.window, len(spread)):
            spread_window = spread.iloc[i - self.window:i+1]
            X_window = spread_window.values.reshape(-1, 1)

            model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=100, random_state=42)
            try:
                model.fit(X_window)
                _, state_probs = model.score_samples(X_window)
            except Exception as e:
                #print(f"Error in HMM computation at index {i}: {e}")
                signals.loc[spread.index[i]] = 0 
                continue # Go to the next iteration

            # Compute the posterior probabilities of the states for the last data point in the window
            current_state_probs = state_probs[-1]
            current_state = np.argmax(current_state_probs)
            current_prob = current_state_probs[current_state]

            # Predicted state probabilities for the next time step
            if self.use_predicted_prob:
                next_state_probs = np.dot(current_state_probs, model.transmat_)
                predicted_state = np.argmax(next_state_probs)
                predicted_prob = next_state_probs[predicted_state]

                current_state = predicted_state
                current_prob = predicted_prob

            state_means_row = pd.DataFrame(
                [model.means_.flatten()],
                index=[spread.index[i]],
                columns=column_names
            )
            self.state_means.append(state_means_row)

            state_vars_row = pd.DataFrame(
                [model.covars_.flatten()],
                index=[spread.index[i]],
                columns=column_names
            )
            self.state_vars.append(state_vars_row)

            state_probs_row = pd.DataFrame(
                [current_state_probs],
                index=[spread.index[i]],
                columns=column_names
            )
            self.state_probs.append(state_probs_row)
            
            # Proceed only if the probability of the current state exceeds the threshold
            if current_prob >= self.state_prob_threshold:
                mu_state = model.means_[current_state][0]
                sigma_state = np.sqrt(model.covars_[current_state][0])

                current_spread = spread.iloc[i]
                z_score = (current_spread - mu_state) / sigma_state

                if position == 0:
                    # Enter long position
                    if z_score < -self.entry_threshold:
                        signals.at[spread.index[i], 'long'] = 1
                        position = 1
                    # Enter short position
                    elif z_score > self.entry_threshold:
                        signals.at[spread.index[i], 'short'] = 1
                        position = -1

                # Exit long position
                elif position == 1 and z_score >= -self.exit_threshold:
                    signals.at[spread.index[i], 'exit'] = 1
                    position = 0

                # Exit short position
                elif position == -1 and z_score <= self.exit_threshold:
                    signals.at[spread.index[i], 'exit'] = 1
                    position = 0

            else:
                # If probability threshold not met, consider exiting any open positions
                if position != 0:
                    signals.at[spread.index[i], 'exit'] = 1
                    position = 0
        
        self.state_means = pd.concat(self.state_means)
        self.state_vars = pd.concat(self.state_vars)
        self.state_probs = pd.concat(self.state_probs)

        return signals.fillna(0)
    

class OrnsteinUhlenbeckStrategy(Strategy):
    def __init__(self, window=50, entry_threshold=1.5, exit_threshold=0.5, use_forecast=False, method='OLS'):
        self.params = {
            'window': window,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'use_forecast': use_forecast,
            'method': method.upper()
        }
        self.initial_params = self.params.copy()
        self.set_params(**self.params)

    def get_params(self):
        return self.params.copy()

    def set_params(self, **params):
        for param in ['window', 'entry_threshold', 'exit_threshold', 'use_forecast', 'method']:
            if param in params:
                self.params[param] = params[param]
                setattr(self, param, params[param])

        # Validation
        if self.window <= 0:
            raise ValueError("Window must be a positive integer.")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("The exit threshold must be less than the entry threshold.")
        if not isinstance(self.use_forecast, bool):
            raise ValueError("use_forecast must be a boolean.")
        if self.method not in ["OLS", "MLE"]:
            raise ValueError("Invalid method specified. Choose 'OLS' or 'MLE'.")

    def reset_params(self):
        self.set_params(**self.initial_params)

    def _estimate_ou_parameters(self, spread_window):        
        if self.method == 'OLS':
            return estimate_ou_parameters_ols(spread_window)
        return estimate_ou_parameters_mle(spread_window)

    def generate_signals(self, spread):
        if self.window >= len(spread):
            raise ValueError("Window size too small, please choose a lower one")
        
        signals = pd.DataFrame(0, index=spread.index, columns=['long', 'short', 'exit'])
        position = 0  # 1 for long, -1 for short, 0 for neutral

        theta_list = []
        mu_list = []
        sigma_list = []

        for i in range(self.window, len(spread)):
            spread_window = spread.iloc[i - self.window:i + 1]
            theta, mu, sigma = self._estimate_ou_parameters(spread_window)

            theta_list.append(theta)
            mu_list.append(mu)
            sigma_list.append(sigma)

            if np.isnan(theta) or np.isnan(mu) or np.isnan(sigma):
                signals.loc[signals.index[i]] = 0
                #print(f"Invalid fitted parameters for OU process. No signal generated for {spread.index[i]}")
                continue  # Skip if parameters are not estimated

            current_spread = spread.iloc[i]
            if self.use_forecast:
                current_spread = current_spread + theta * (mu - current_spread)
            z_score = (current_spread - mu) / sigma

            if position == 0:
                # Enter long position
                if z_score < -self.entry_threshold:
                    signals.at[signals.index[i], 'long'] = 1
                    position = 1

                # Enter short position
                elif z_score > self.entry_threshold:
                    signals.at[signals.index[i], 'short'] = 1
                    position = -1

            # Exit long position
            elif position == 1 and z_score > -self.exit_threshold:
                signals.at[signals.index[i], 'exit'] = 1
                position = 0

            # Exit short position
            elif position == -1 and z_score < self.exit_threshold:
                signals.at[signals.index[i], 'exit'] = 1
                position = 0

        self.theta_series = pd.Series([np.nan]*self.window + theta_list, index=spread.index)
        self.mu_series = pd.Series([np.nan]*self.window + mu_list, index=spread.index)
        self.sigma_series = pd.Series([np.nan]*self.window + sigma_list, index=spread.index)

        return signals.fillna(0)