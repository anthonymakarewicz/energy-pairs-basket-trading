import numpy as np
import pandas as pd
import statsmodels.api as sm

from abc import ABC, abstractmethod
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


class HedgeRatio(ABC):
    def __init__(self, method='dynamic', window=20, **kwargs):
        self.supported_methods = ['static', 'dynamic']
        if method not in self.supported_methods:
            raise ValueError(f"{self.__class__.__name__} does not support method '{method}'."
                             f"Supported methods: {self.supported_methods}")
        
        self.method = method
        self.window = window
        self.params = {'method': self.method, 'window': self.window}
        self.initial_params = self.params.copy()
        self.set_params(**kwargs)

    def set_params(self, **params):
        # Update method if provided
        if 'method' in params:
            if params['method'] not in self.supported_methods:
                raise ValueError(f"{self.__class__.__name__} does not support method '{params['method']}'."
                                 f"Supported methods: {self.supported_methods}")
            self.method = params['method']
            self.params['method'] = self.method

        if 'window' in params:
            self.window = params['window']
            self.params['window'] = self.window

        if self.method == 'dynamic':
            if not isinstance(self.window, int) or self.window <= 0:
                raise ValueError("Window must be a positive integer.")

    def get_params(self):
        return self.params.copy()

    def reset_params(self):
        self.set_params(**self.initial_params)

    @abstractmethod
    def compute_hedge_ratio(self, data, assets):
        pass


class KalmanFilterHedgeRatio(HedgeRatio):
    def __init__(self, kalman_model):
        self.kalman_model = kalman_model
        self.method = "dynamic"
        self.params = {}
        self.initial_params = self.params.copy()

    def get_params(self):
        return self.params.copy()

    def set_params(self, **params):
        if 'delta' in params:
            delta = params['delta']
            if delta != self.kalman_model.params["delta"]:
                self.kalman_model.set_params(delta=delta)

    def reset_params(self):
        self.set_params(**self.initial_params)
        self.kalman_model.reset_params()

    def compute_hedge_ratio(self, data, assets):
        return self.kalman_model.get_hedge_ratios()
    

class OLSHedgeRatio(HedgeRatio):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_params = self.params.copy()
        
    def compute_hedge_ratio(self, data, assets):
        y = data[assets[0]]
        x = data[assets[1:]]

        if self.method == "dynamic":
            hedge_ratios = []
            for i in range(self.window, len(data)):
                y_window = y.iloc[i-self.window:i+1]
                x_window = x.iloc[i-self.window:i+1]

                X = sm.add_constant(x_window)
                model = sm.OLS(y_window, X).fit()
                weights = model.params[1:]
                hedge_ratios.append([1] + (-weights).to_list())

            hedge_ratios = pd.DataFrame(hedge_ratios,
                                        index=data.index[self.window:],
                                        columns=assets)
            hedge_ratios = hedge_ratios.reindex(data.index, fill_value=np.nan)
        else:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            weights = model.params[1:]
            hedge_ratios = pd.DataFrame([1] + (-weights).to_list(), 
                                        index=assets,
                                        columns=[data.index[-1]]).T
   
        return hedge_ratios
    

class DoubleExponentialSmoothingHedgeRatio(HedgeRatio):
    def __init__(self, smoothing_level=0.2, smoothing_trend=0.2, **kwargs):
        self.supported_methods = ['dynamic']
        super().__init__(method='dynamic', **kwargs)
        self.alpha = smoothing_level
        self.beta = smoothing_trend
        self.params.update({'smoothing_level': self.alpha, 'smoothing_trend': self.beta})
        self.initial_params = self.get_params()

    def set_params(self, **params):
        if 'method' in params and params['method'] != 'dynamic':
            raise ValueError(f"{self.__class__.__name__} only supports 'dynamic' method.")
        super().set_params(**params)

        if 'smoothing_level' in params:
            self.alpha = params['smoothing_level']
            self.params['smoothing_level'] = self.alpha
        if 'smoothing_trend' in params:
            self.beta = params['smoothing_trend']
            self.params['smoothing_trend'] = self.beta

    def compute_hedge_ratio(self, data, assets):
        if len(assets) > 2:
            raise ValueError("DoubleExponentialSmoothing not available for more than 2 assets")
        
        hedge_ratios = pd.Series(index=data.index, name=assets[1])
        ratio = data[assets[0]] / data[assets[1]]

        # Initialize level and trend
        L_prev = ratio.iloc[0]
        T_prev = ratio.iloc[1] - ratio.iloc[0]
        hedge_ratios.iloc[0] = L_prev + T_prev  # Forecast for time t = 1

        for t in range(1, len(ratio)):
            y_t = ratio.iloc[t]
            L_t = self.alpha * y_t + (1 - self.alpha) * (L_prev + T_prev)
            T_t = self.beta * (L_t - L_prev) + (1 - self.beta) * T_prev

            # Forecast for the next time step
            hedge_ratios.iloc[t] = L_t + T_t

            L_prev = L_t
            T_prev = T_t
            
        hedge_ratios = pd.concat([pd.Series(1, index=data.index, name=assets[0]), 
                                  -hedge_ratios], axis=1)
        return hedge_ratios
    

class JohansenHedgeRatio(HedgeRatio):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_hedge_ratio(self, data, assets):
        data = data[assets]
        hedge_ratios = []

        if self.method == "dynamic":
            for i in range(self.window, len(data)):
                data_window = data.iloc[i-self.window:i+1]
                result = coint_johansen(data_window, det_order=1, k_ar_diff=1)
                hedge_ratio = result.evec[:, 0]
                hedge_ratios.append(hedge_ratio)
                
            hedge_ratios = pd.DataFrame(hedge_ratios,
                                        index=data.index[self.window:],
                                        columns=assets).reindex(data.index, fill_value=np.nan)
        else:
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            hedge_ratio = result.evec[:, 0]
            hedge_ratios = pd.DataFrame(hedge_ratio, index=assets, columns=[data.index[-1]]).T

        return hedge_ratios 


#############################
# Experimental Hedge Ratios #
#############################

class ElasticNetHedgeRatio(HedgeRatio):
    def __init__(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha         # Regularization strength
        self.l1_ratio = l1_ratio  # Balance between L1 and L2 regularization
        self.params = {'alpha': self.alpha, 'l1_ratio': self.l1_ratio}

    def compute_hedge_ratio(self, data, assets):
        y = data[assets[0]]
        X = data[assets[1:]]
        
        if self.method == "dynamic":
            hedge_ratios = []
            for i in range(self.window, len(data)):
                y_window = y.iloc[i - self.window:i+1]
                X_window = X.iloc[i - self.window:i+1]

                model = ElasticNet(**self.params, fit_intercept=False)
                model.fit(X_window, y_window)
                weights = model.coef_
                hedge_ratios.append([1] + (-weights).tolist())

            hedge_ratios = pd.DataFrame(hedge_ratios,
                                        index=data.index[self.window:],
                                        columns=assets)
            hedge_ratios = hedge_ratios.reindex(data.index, fill_value=np.nan)
        else:
            model = ElasticNet(**self.params, fit_intercept=False)
            model.fit(X, y)
            weights = model.coef_
            hedge_ratios = pd.DataFrame([1] + (-weights).tolist(), 
                                        index=assets, 
                                        columns=[data.index[-1]]).T

        return hedge_ratios
    
    
class TLSHedgeRatio(HedgeRatio):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_params = self.params.copy()
        
    def compute_hedge_ratio(self, data, assets):
        """
        Compute the hedge ratios using Total Least Squares (TLS).

        Parameters:
        - data: pd.DataFrame containing price data.
        - assets: List of asset names.

        Returns:
        - hedge_ratios: pd.DataFrame of hedge ratios.
        """
        y = data[assets[0]].values.reshape(-1, 1)
        X = data[assets[1:]].values
        
        if self.method == "dynamic":
            if self.window > len(data):
                raise ValueError("Window size is larger than the data length.")
            
            hedge_ratios = []
            for i in range(self.window, len(data)):
                y_window = y[i - self.window:i]
                X_window = X[i - self.window:i]

                if np.isnan(X_window).any() or np.isnan(y_window).any():
                    print(f"NaN values detected at index {i}, skipping this window.")
                    continue

                try:
                    Z = np.hstack((X_window, y_window))
                    _, _, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
                    V = Vt.T

                    # Normalize so that the coefficient of y is 1
                    tls_solution = V[:, -1]
                    tls_solution = tls_solution / tls_solution[-1]

                    # Extract the hedge ratio coefficients
                    hedge_ratio = tls_solution[:-1]
                    hedge_ratios.append([1] + (-hedge_ratio).tolist())
                except np.linalg.LinAlgError as e:
                    print(f"Numerical issue at index {i}: {e}")
                    continue

            hedge_ratios = pd.DataFrame(
                hedge_ratios,
                index=data.index[self.window:],
                columns=assets
            ).reindex(data.index, fill_value=np.nan)
        else:
            Z = np.hstack((X, y))
            _, _, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
            V = Vt.T

            # Normalize so that the coefficient of y is 1
            tls_solution = V[:, -1]
            tls_solution = tls_solution / tls_solution[-1]
            hedge_ratio = tls_solution[:-1]
            hedge_ratios = pd.DataFrame(
                [1] + (-hedge_ratio).tolist(),
                index=assets,
                columns=[data.index[-1]]
            ).T

        return hedge_ratios
    

class PLSHedgeRatio(HedgeRatio):
    def __init__(self, n_components=1, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def compute_hedge_ratio(self, data, assets):
        y = data[assets[0]].values.reshape(-1, 1)
        X = data[assets[1:]]
        
        if self.method == "dynamic":
            hedge_ratios = []
            for i in range(self.window, len(data)):
                y_window = y[i - self.window:i+1]
                X_window = X.iloc[i - self.window:i+1]

                model = PLSRegression(n_components=self.n_components)
                model.fit(X_window, y_window)
                weights = model.coef_.flatten()
                hedge_ratios.append([1] + (-weights).tolist())

            hedge_ratios = pd.DataFrame(hedge_ratios,
                                        index=data.index[self.window:],
                                        columns=assets)
            hedge_ratios = hedge_ratios.reindex(data.index, fill_value=np.nan)
        else:
            model = PLSRegression(n_components=self.n_components)
            model.fit(X, y)
            weights = model.coef_.flatten()
            hedge_ratios = pd.DataFrame([1] + (-weights).tolist(), 
                                        index=assets, 
                                        columns=[data.index[-1]]).T

        return hedge_ratios


class PCAHedgeRatio(HedgeRatio):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_params = self.params.copy()

    def compute_hedge_ratio(self, data, assets):
        data = data[assets]
        
        if self.method == "dynamic":
            hedge_ratios = []
            index = []
            for i in range(self.window, len(data)):
                curr_data = data.iloc[i-self.window:i+1]

                pca = PCA(n_components=len(assets))
                pca.fit(curr_data)

                weights = pca.components_[0]
                hedge_ratios.append(weights)
                index.append(data.index[i])
                    
            hedge_ratios = pd.DataFrame(hedge_ratios,
                                        index=index,
                                        columns=assets)
            hedge_ratios = hedge_ratios.reindex(data.index, fill_value=np.nan)
        else:
            pca = PCA(n_components=1)
            pca.fit(data)
            weights = pca.components_[0]
            hedge_ratios = pd.DataFrame([weights],
                                        index=[data.index[-1]],
                                        columns=assets)
            hedge_ratios = hedge_ratios.reindex(data.index, fill_value=np.nan)

        return hedge_ratios
    

class VECMHedgeRatio(HedgeRatio):
    def __init__(self, lag_order=1, coint_rank=1, **kwargs):
        super().__init__(**kwargs)
        self.lag_order = lag_order
        self.coint_rank = coint_rank

    def compute_hedge_ratio(self, data, assets):
        data = data[assets]
        hedge_ratios = []

        if self.method == "dynamic":
            for i in range(self.window, len(data)):
                data_window = data.iloc[i - self.window : i]
                model = VECM(
                    endog=data_window.diff().dropna(), 
                    k_ar_diff=self.lag_order, 
                    coint_rank=self.coint_rank
                )
                vecm_result = model.fit()
                hedge_ratio = vecm_result.alpha[:, 0]
                hedge_ratios.append(hedge_ratio)
                
            hedge_ratios = pd.DataFrame(hedge_ratios,
                                        index=data.index[self.window:],
                                        columns=assets)
            hedge_ratios = hedge_ratios.reindex(data.index, fill_value=np.nan)
        else:
            model = VECM(
                endog=data.diff().dropna(), 
                k_ar_diff=self.lag_order, 
                coint_rank=self.coint_rank
            )
            vecm_result = model.fit()
            hedge_ratio = vecm_result.alpha[:, 0]
            hedge_ratios = pd.DataFrame([hedge_ratio],
                                        index=[data.index[-1]],
                                        columns=assets)
            hedge_ratios = hedge_ratios.reindex(data.index, fill_value=np.nan)
        return hedge_ratios