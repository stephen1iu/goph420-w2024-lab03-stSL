import numpy as np

def multi_regress(y, Z):
    """
    Performs multiple linear regression.

    Parameters
    ----------
    y : array_like, shape of (n,) or (n,1)
    The vector of dependent variable data

    Z : array_like, shape of (n,m)
    The matrix of independent variable data
    
    Returns
    -------
    a: numpy.ndarray, shape of (m,) or (m,1)
    The vector of model coefficients
    
    e: numpy.ndarray, shape of (n,) or (n,1)
    The vector of residuals
    
    r_sq: float
    The coefficient of determination, r^2 value
    """

    Z_t = np.transpose(Z)
    a = np.dot(np.linalg.inv(Z_t @ Z), (Z_t @ y)) 

    y_aprx = Z @ a
    e = y - y_aprx

    ssr = np.dot(np.transpose(e), e)
    avg_y = np.full_like(y, np.mean(y))
    e_avg = y - avg_y #y - y_mean for each row
    sst = np.dot(np.transpose(e_avg), e_avg)

    r_sq = np.divide((sst - ssr), sst)

    return a, e, r_sq