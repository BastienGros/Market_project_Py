import pandas as pd 
import yfinance as yf
import numpy as np
import seaborn as sb
import matplotlib.pyplot as mpl
from dataclasses import dataclass
import math
from scipy import stats

# In this file, we want to simulate the heston model to find the volatility of the underlying asset


def heston_model_sim(S0, v0, rho, kappa, theta, sigma,T, N, M):
    """
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - T     : time of simulation
     - N     : number of time steps
     - M     : number of scenarios / simulations
    
    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = T/N
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])
    
    # arrays to store prices and variances
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)
    
    # sampling correlated brownian motions under risk-neutral measure
    
    Z = np.random.multivariate_normal(mu, cov, (N,M))
    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0])
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)
    
    return S, v

# Parameters

S0 = 100.0             # asset price
T = 1.0                # time in years
r = 0.02               # risk-free rate
Nsteps = 252           # number of time steps in simulation
Msimulations = 1000    # number of simulations

# Heston dependent parameters

kappa = 3              # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.20**2        # long-term mean of variance under risk-neutral dynamics
v0 = 0.25**2           # initial variance under risk-neutral dynamics
rho = 0.7              # correlation between returns and variances under risk-neutral dynamics
sigma = 0.6            # volatility of volatility

rho_p = 0.98
rho_n = -0.98

S_p,v_p = heston_model_sim(S0, v0, rho_p, kappa, theta, sigma,T, Nsteps, Msimulations)
S_n,v_n = heston_model_sim(S0, v0, rho_n, kappa, theta, sigma,T, Nsteps, Msimulations)

mpl.subplot(1,2,1)
time = np.linspace(0,T,Nsteps+1)

mpl.plot(time,S_p)
mpl.title('Heston Model Asset Prices')
mpl.xlabel('Time')
mpl.ylabel('Asset Prices')

mpl.subplot(1,2,2)
mpl.plot(time,v_p)
mpl.title('Heston Model Variance Process')
mpl.xlabel('Time')
mpl.ylabel('Variance')
mpl.show()

Geometric_brownian_motion  = S0*np.exp((r - theta**2/2)*T + np.sqrt(T)*np.random.normal(0,1,Msimulations))

mpl.subplot()
sb.kdeplot(S_p[-1], label=r"$\rho= 0.98$")
sb.kdeplot(S_n[-1], label=r"$\rho= -0.98$")
sb.kdeplot(Geometric_brownian_motion, label="GBM")

mpl.title(r'Asset price density under Heston model')
mpl.xlim([20, 180])
mpl.xlabel('$S_T$')
mpl.ylabel('Density')
mpl.legend()
mpl.show()