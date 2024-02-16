import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as mpl
from dataclasses import dataclass
from scipy import stats as loi
import yfinance as yf


def CallPutPricing(s,k,r,t,vol, Type):
    n1 = (math.log(s/k)+(r+((vol)**2)/2)*t)/vol*math.sqrt(t)
    n2 = n1 - vol*math.sqrt(t)
    
    PriceCallPut = Type*s*loi.norm.cdf(Type*n1)-Type*k*math.exp((-r)*t)*loi.norm.cdf(n2)
    return PriceCallPut

## implied volatility calculed with the newton-raphson algorithm ## 

def vol_Implicite(s,k,r,t,vol, Type, tol = 0.000001):
    
    max_iter = 1000
    vol_old = 0.30
    count = 0
    epsilon = 1
    
    while epsilon > tol:
        count += 1
        if count >= max_iter:
            print("Breakpoint reach")
            break
        
        origi_vol = vol_old
        function_value = CallPutPricing(s,k,r,t,vol, Type)
        
        n1 = (math.log(s/k)+(r+((vol)**2)/2)*t)/vol*math.sqrt(t)
             
        vega = s*loi.norm.pdf(n1)*math.sqrt(t)
        vol_old = -function_value/vega+vol_old
    
    epsilon = abs((vol - origi_vol)/origi_vol)
    print(vol_old)


@dataclass
class option_chain__return_Values :
    DataFrame: pd.DataFrame
    Date: str
    Ticker: str 
    type: int   
    

def GetOptionsChain(ticker):
    var = yf.Ticker(ticker)
    str = ticker
    Info_Options = var.options
    print(Info_Options)
    
    date = input("Which date do you want to be informed of ? Write it in the following format : YYYY-MM-DD") 
    opt = var.option_chain(date)
    CallPut = input("CALL OR PUT ?")
    
    if CallPut == "CALL":
        
         call = opt.calls
         Data_return = option_chain__return_Values(call, date, str, 1)
         return Data_return
     
    else :
        put = opt.puts
        Data_return = option_chain__return_Values(put, date, str, -1)
        return Data_return
        
def LastStockPrice(ticker):
    
    select_ticker = yf.Ticker(ticker) 
    data = select_ticker.history()
    last_quote = data['close'].iloc[-1]
    return last_quote 
        
        
    
def VolumeFilterOnOption(file : pd.DataFrame):
    Workfile = file[file['volume']>100]
    return Workfile

    
msft1 = GetOptionsChain("MSFT")
msft2 = GetOptionsChain("MSFT")
msft3 = GetOptionsChain("MSFT")

NewMsft1 = VolumeFilterOnOption(msft1.DataFrame)
NewMsft2 = VolumeFilterOnOption(msft2.DataFrame)
NewMsft3 = VolumeFilterOnOption(msft3.DataFrame)


print(NewMsft1)

mpl.plot(msft1.DataFrame['strike'], msft1.DataFrame['impliedVolatility']*100, label= msft1.Date)
mpl.plot(msft2.DataFrame['strike'], msft2.DataFrame['impliedVolatility']*100, label= msft2.Date)
mpl.plot(msft3.DataFrame['strike'], msft3.DataFrame['impliedVolatility']*100, label= msft3.Date)

mpl.legend()
mpl.show()

## Volatility smile ##
## As we can see, the implied volatility decrease when the strike get closer to the actual price of the asset and increase when the strike move away from it  ##
## The options ATM have an smaller implied volatility than OTM and ITM options ##
## The reason for the volatility smile is undoubtedly the fact that, on the equity markets, most investors have long positions. 
## If they want to protect their positions using strategies involving options, they tend to use two common hedging strategies:
## The protective put and the covered call. But in most cases, buying a protective put involves buying a put that is not ITM
## And selling a covered call involves selling a call that is not ITM.

## Term structure ## 
## Also, it is important to note regarding the term structure that the closer the option's expiration,
## the more the volatility smile tends to flatten. 
## This can be explained by the fact that the option has much less chance of returning to the money as it approaches expiration.




