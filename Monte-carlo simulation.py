import pandas as pd 
import yfinance as yf
import numpy as np
import matplotlib.pyplot as mpl
from dataclasses import dataclass
import math
from scipy import stats

@dataclass
class Stock_return_fonction :
    Frame : pd.DataFrame
    ticker : str
    beginning : str
    end : str

class stock : 
    
    def __init__(self, Ticker, Start, End):
        self.ticker = Ticker
        self.start = Start
        self.end = End
        self.Stock_return = stock.stockHistory(self, self.ticker, self.start, self.end)
        self.log_return = stock.Log_return(self,self.Stock_return)
        
        

    def stockHistory(self,ticker, Start, End):
    # Download the stock data
     Data = yf.download(ticker, start=Start, end=End)
    # We draw the close of the stock in the period

     mpl.figure(figsize=(10,6))
     mpl.plot(Data['Close'], label=f'The price of {ticker}')
     mpl.title(f'Price evolution of {ticker} from {Start} to {End}')
     mpl.xlabel('Date')
     mpl.ylabel('Closing price')
     mpl.legend()
     mpl.grid(True)
     mpl.show()
     Return = Stock_return_fonction(Data, ticker, Start, End)
     return Return


    
    def Log_return(self,Data : Stock_return_fonction) :
    
    #Calculate the log return of the stock
     Data.Frame['log_Return'] = Data.Frame['Close'].pct_change().apply(lambda x: math.log(1+x))
    
     mpl.figure(figsize=(10,6))
     mpl.plot(Data.Frame['log_Return'], label=f'Log Returns of {Data.ticker}')
     mpl.title(f'Log return of {Data.ticker} from {Data.beginning} to {Data.end}')
     mpl.xlabel('Date')
     mpl.ylabel('Log Returns')
     mpl.legend()
     mpl.grid(True)
     mpl.show()
    
    def Monte_Carlo_GBM(self,S0 : pd.DataFrame, mu, sigma, dt, n):
        """
        Generate Monte Carlo paths for a geometric brownien movement(GBM)
    
        Parameters : 
    
         - S0 : Initial price of the underlying asset
        - mu : Average return (drift)
        - sigma : Volatility
        - dt : Option Maturity
        - n : Number of steps
    
        Return : 
    
        - x : The monte-carlo paths array
    
     """
        S0 = S0['log_Return'].iloc[-1]
        
        x = np.exp((mu - sigma**2/2)*dt + sigma * np.random.normal(0,np.sqrt(dt), size=(len(sigma), n)).T)
        x = np.vstack([np.ones(len(sigma)), x])
        x = S0 * x.cumprod(axis=0)
            
        
        mpl.figure(figsize=(10,6))
        mpl.plot(x)         
        mpl.title(f'Monte-Carlo simulation for {self.ticker}')
        mpl.xlabel('Time') 
        mpl.ylabel('Underlying price')
        mpl.legend()
        mpl.grid(True)
        mpl.show()  
    
        return x
    
    @property
    def Get_Stock_return(self):
        return self.Stock_return
       

essai = stock('MSFT', '2023-01-01', '2023-12-31') 
    
# Testing DATA
    
mu = 0.01
sigma = np.arange(0, 5, 0.2)
T = 1.0
n = 50
dt = 0.1


    
# We generate the Monte-Carlo paths 
    
St = stock.Monte_Carlo_GBM(essai, essai.Get_Stock_return.Frame, mu, sigma,dt , n)
        
# We can see that using random value for Mu and Sigma in the GBM model isn't usually recommanded because it doesn't reflect 
# the financial market reality. The Mu (average return on stock or drift) and Sigma (volatility) parameters are important 
# financial assets caracteristics and their estimation from historical data are overall prefered.

# The most common approch consist in using historical data to estimate mu and sigma. It can be done by calculating the average return 
# and the standar error deviation on a specific period. By using those estimations, you can simulate the price movement thanks to the 
# GBM model

Average_mu = np.mean(essai.Get_Stock_return.Frame['log_Return']) 
Average_Sigma = np.std(essai.Get_Stock_return.Frame['log_Return'])







    
