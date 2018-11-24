import numpy as np
import math
from scipy.stats import norm


def logreturn_to_return(log_return):
    return math.exp(log_return) - 1


def return_to_logreturn(ret):
    return np.log(1 + ret)


def value_at_risk(confidence_interval, mean, std, value=1):
    alpha = norm.ppf(1-confidence_interval, mean, std)
    return value * logreturn_to_return(alpha)
