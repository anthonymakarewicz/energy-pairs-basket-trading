import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.optimize import minimize
import statsmodels.api as sm


class KalmanFilterModel:
    def __init__(self, data, delta=1e-5):
        self.data = data
        self.assets = data.columns
        self.params = {
            'delta': delta
        }
        self.delta = delta
        self.initial_params = self.params.copy()
        self.set_params(**self.params)

        # Cache
        self.kf = None
        self.hedge_ratios = None
        self.prediction_errors = None
        self.prediction_variances = None

    def fit(self):
        y = self.data[self.assets[0]].values   # Asset 1 prices (y_t)
        x = self.data[self.assets[1]].values   # Asset 2 prices (x_t)
        obs_mat = x.reshape(-1, 1, 1)          # Observation matrices (H_t)

        # Process noise parameter
        trans_cov =  self.delta/ (1 - self.delta) * np.eye(1)  # Process noise covariance (Q)

        # Initialize the Kalman Filter
        self.kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=1,
            initial_state_mean=0,
            initial_state_covariance=1,
            transition_matrices=1,             # State transition matrix (F_t = 1)
            observation_matrices=obs_mat,      # Observation matrices (H_t = x_t)
            observation_covariance=1.0,        # Observation noise covariance (R)
            transition_covariance=trans_cov    # Process noise covariance (Q)
        )

        # Perform Kalman filtering
        state_means, state_covariances = self.kf.filter(y)
        
        # Compute prior estimates for t+1
        beta_prior = state_means[:-1]  # beta_{t+1|t} = beta_{t|t}
        P_prior = state_covariances[:-1] + self.kf.transition_covariance  # P_{t+1|t} = P_{t|t} + Q

        # Observations at t+1
        y_post = y[1:]  # y_{t+1}
        x_post = x[1:]  # x_{t+1}

        # Predicted observations: E[y_{t+1} | t] = x_{t+1} * beta_{t+1|t}
        predicted_obs = x_post * beta_prior[:, 0]

        # Prediction errors: y_{t+1} - E[y_{t+1} | t]
        prediction_errors = y_post - predicted_obs

        # Prediction variances: Var[y_{t+1} | t] = x_{t+1}^2 * P_{t+1|t} + R
        prediction_variances = (x_post ** 2) * P_prior[:, 0, 0] + self.kf.observation_covariance

        # Cache the results
        self.hedge_ratios = pd.DataFrame(-state_means.flatten(), index=self.data.index, columns=[self.assets[1]])
        self.hedge_ratios[self.assets[0]] = 1
        self.hedge_ratios = self.hedge_ratios[self.assets]
        self.prediction_errors = pd.Series(prediction_errors, index=self.data.index[1:])
        self.prediction_variances = pd.Series(prediction_variances, index=self.data.index[1:])

    def get_params(self):
        return self.params.copy()

    def set_params(self, **params):
        delta = params.get('delta', self.delta)
        if delta != self.params.get("delta", None):
            self.params["delta"] = delta
            self.delta = delta
            # Clear cached results and re-fit the model
            self.kf = None
            self.hedge_ratios = None
            self.prediction_errors = None
            self.prediction_variances = None
            self.fit()

    def reset_params(self):
        self.set_params(**self.initial_params)

    def get_hedge_ratios(self):
        if self.hedge_ratios is None:
            self.fit()
        return self.hedge_ratios

    def get_prediction_errors(self):
        if self.prediction_errors is None:
            self.fit()
        return self.prediction_errors

    def get_prediction_variances(self):
        if self.prediction_variances is None:
            self.fit()
        return self.prediction_variances
    

def estimate_ou_parameters_ols(spread_window):
    Y = spread_window[1:].values    # X_{t+1}
    X = spread_window[:-1].values  # X_t
    X = sm.add_constant(X)        # Adds a constant term to the predictor

    model = sm.OLS(Y, X)
    results = model.fit()

    a = results.params[1]
    b = results.params[0]
    residuals = results.resid
    sigma = np.std(residuals)

    theta = mu = np.nan
    if a > 0 and a != 1:
        theta = -np.log(a)
        mu = b / (1 - a)
        
    return theta, mu, sigma


def estimate_ou_parameters_mle(spread):     
    theta_init = 0.1
    mu_init = spread.mean()
    sigma_init = spread.std()

    X = spread.values
    n = len(X) - 1   

    def neg_log_likelihood(params):
        theta, mu, sigma = params
        if theta <= 0 or sigma <= 0:
            return np.inf  # Ensure positive parameters

        # Calculate m_t, V_t and the residuals (dt = 1)
        e_neg_theta = np.exp(-theta)
        m_t = mu + (X[:-1] - mu) * e_neg_theta
        V_t = (sigma**2) / (2 * theta) * (1 - e_neg_theta**2)
        residuals = X[1:] - m_t

        # Compute the negative log-likelihood
        nll = 0.5 * (n * np.log(2*np.pi) + np.sum(np.log(V_t) + (residuals**2) / V_t))
        return nll

    params_init = [theta_init, mu_init, sigma_init]
    bounds = [(1e-5, None), (None, None), (1e-5, None)]  # theta > 0, mu unbounded, sigma > 0

    # Minimize the negative log-likelihood
    result = minimize(neg_log_likelihood, params_init, bounds=bounds, method='L-BFGS-B')

    if result.success:
        theta, mu, sigma = result.x
        return theta, mu, sigma
    else:
        print(f"Optimization failed: {result.message}", ". Parameters set to None")