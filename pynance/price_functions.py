
import pandas as pd
import numpy as np
import warnings
import math
import scipy.optimize as sco
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from utils import logreturn_to_return

class PricesDataFrame():
    def __init__(self, prices):
        if not isinstance(prices, pd.DataFrame):
            warnings.warn("Prices are not in a dataframe", RuntimeWarning)
        prices.sort_index(inplace=True)
        self.prices = prices

    def returns(self):
        df_return = pd.DataFrame()
        for col in self.prices.columns:
            df_return[col] = np.log(self.prices[col]) - np.log(self.prices[col].shift(1))
        return df_return.dropna()

    def mean_returns(self):
        return pd.DataFrame(self.returns().mean(), columns=['mean_returns'])

    def variance(self):
        return pd.DataFrame(self.returns().var().apply(math.sqrt), columns=['variance'])

    def correlation(self):
        return self.returns().corr()

    def portfolio_return(self, weights):
        weights = pd.DataFrame(weights)
        return pd.DataFrame(self.mean_returns().values*weights.values, columns=['portfolio_return'], index=weights.index).sum()[0]

    def portfolio_volatility(self, weights):
        #Gambiarra:
        weights = pd.DataFrame(weights).values
        return math.sqrt(weights.T.dot(self.returns().cov().dot(weights)))

    def portfolio_sharpe(self, weights, risk_free_rate, otimizer_ajust=False):
        #Gambiarra:
        weights = pd.DataFrame(weights).values
        result = (self.portfolio_return(weights) - risk_free_rate) / self.portfolio_volatility(weights)
        return result * -1 if otimizer_ajust else result

    def portfolio_efficient_return(self, return_target, use_negative=True):
        assets = self.mean_returns().index
        num_assets = len(assets)
        constraints = ({'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - return_target},
               {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for asset in range(num_assets))
        bound = (-1.0,1.0) if use_negative else (0,1.0)
        bounds = tuple(bound for asset in range(num_assets))
        x0 = pd.DataFrame(num_assets * [1. /num_assets, ], index=assets)

        otimizer_result = sco.minimize(self.portfolio_volatility, x0,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        weights = pd.DataFrame(otimizer_result['x'], index=assets, columns=['weights'])
        result = {'weights': weights,
        'portfolio_return': self.portfolio_return(weights),
        'portfolio_volatility': self.portfolio_volatility(weights)}
        return result

    def min_volatility(self, use_negative=True):
        assets = self.mean_returns().index
        num_assets = len(assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        #passar bound como parametro
        bound = (-1.0,1.0) if use_negative else (0,1.0)
        bounds = tuple(bound for asset in range(num_assets))
        x0 = pd.DataFrame(num_assets * [1. /num_assets, ], index=assets)

        otimizer_result = sco.minimize(self.portfolio_volatility, x0,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        weights = pd.DataFrame(otimizer_result['x'], index=assets, columns=['weights'])
        result = {'weights': weights,
        'portfolio_return': self.portfolio_return(weights),
        'portfolio_volatility': self.portfolio_volatility(weights)}

        return result

    def max_sharpe(self, risk_free_rate, use_negative=True):
        assets = self.mean_returns().index
        num_assets = len(assets)
        otimizer_ajust=True
        args = (risk_free_rate, otimizer_ajust)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (-1.0,1.0) if use_negative else (0,1.0)
        bounds = tuple(bound for asset in range(num_assets))
        x0 = pd.DataFrame(num_assets * [1. /num_assets, ], index=assets)

        otimizer_result = sco.minimize(self.portfolio_sharpe, x0, args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        weights = pd.DataFrame(otimizer_result['x'], index=assets, columns=['weights'])
        result = {'weights': weights,
        'portfolio_return': self.portfolio_return(weights),
        'portfolio_volatility': self.portfolio_volatility(weights)}
        return result

    def efficient_frontier(self, risk_free_rate, use_negative=True):
        min_volatility = self.min_volatility(use_negative=use_negative)
        max_sharpe = self.max_sharpe(use_negative=use_negative,risk_free_rate=risk_free_rate)
        ts = np.linspace(-2, 2, 100)
        frontier = []
        for t in ts:
            weights = (t* min_volatility['weights']) + ((1-t) * max_sharpe['weights'])
            frontier.append([self.portfolio_return(weights), self.portfolio_volatility(weights),weights])
        return pd.DataFrame(frontier, columns=['portfolio_return', 'portfolio_volatility', 'weights'])


    def plot_efficient_frontier(self, risk_free_rate, use_negative=True):
        min_volatility = self.min_volatility(use_negative=use_negative)
        max_sharpe = self.max_sharpe(use_negative=use_negative,risk_free_rate=risk_free_rate)
        df_frontier = self.efficient_frontier(use_negative=use_negative,risk_free_rate=risk_free_rate)
        variance = self.variance()
        mean_returns = self.mean_returns()
        fig= plt.figure(figsize=(16,10))
        plt.plot( 'portfolio_volatility', 'portfolio_return', data=df_frontier, color='black')
        plt.scatter(max_sharpe['portfolio_volatility'], max_sharpe['portfolio_return'], marker='X',color='r',s=200, label='Maximum Sharpe')
        plt.scatter(min_volatility['portfolio_volatility'], min_volatility['portfolio_return'],marker='X',color='g',s=200, label='Minimum Volatility')
        for asset in mean_returns.index:
            plt.scatter(variance.loc[asset], mean_returns.loc[asset],marker='o',color='b',s=100)
            plt.annotate(asset,
                     xy=(self.variance().loc[asset], self.mean_returns().loc[asset]),
                     xytext=(10, 0),
                     textcoords='offset points')

        plt.legend(['Efficient frontier', 'Maximum Sharpe', 'Minimum Volatility'])
        plt.show()


    def portfolio_return_for_row(self, weights):
        list_returns = []
        returns = self.returns()
        for index in returns.index:
            ret = weights.T.dot(returns.loc[index])
            list_returns.append(ret.values)
        return pd.DataFrame(list_returns, index=returns.index, columns=['portfolio_return'])

    def portfolio_cumulative_return(self, weights):
        return self.portfolio_return_for_row(weights).cumsum()

    def portfolio_value_for_row(self, weights, inicial_value):
        return (self.portfolio_return_for_row(weights).cumsum().applymap(logreturn_to_return) + 1) * inicial_value


    def portfolio_return_for_all_period(self, weights):
                list_returns = []
                returns = self.returns()
                for index in returns.index:
                    ret = weights.T.dot(returns.loc[index])
                    list_returns.append(ret.values)
                return pd.DataFrame(list_returns, index=returns.index, columns=['portfolio_return']).sum()
