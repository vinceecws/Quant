from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from pandas import Series

class StationaryTests():

	@staticmethod
	def adf_test(series):
	    print('Dickey-Fuller Test:')
	    #TESTS DIFFERENCE-STATIONARITY
		#Null hypothesis: Unit-root (non-stationary/difference-stationary)
		#Alternative hypothesis: No unit-root (stationary/trend-stationary)
	    test = adfuller(series)
	    output = Series(test[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	    for key, value in test[4].items():
	            output['Critical Value (%s)'%key] = value
	    print(output)
	    if(test[0]<test[4]['1%']):
	        print('Passed ADF @ 1%')
	    elif(test[0]<test[4]['5%']):
	        print('Passed ADF @ 5%')
	    elif(test[0]<test[4]['10%']):
	        print('Passed ADF @ 10%')
	    else:
	        print('Failed ADF: Non-stationary (Difference)')

	@staticmethod
	def kpss_test(series): 
		#TESTS TREND-STATIONARITY, TWO-TAILED HYPOTHESIS TEST
		#Null hypothesis: No unit-root around a deterministic trend (stationary/difference-stationary)
		#Alternative hypothesis: Unit-root around a trend (non-stationary/trend-stationary)
	    test = kpss(series, regression='c')
	    output = Series(test[0:3], index=['Test Statistic','p-value','Lags Used'])
	    for key, value in test[3].items():
	            output['Critical Value (%s)'%key] = value
	    print(output)
	    if(abs(test[0])<abs(test[3]['10%'])):
	        print('Passed KPSS @ 10%')
	    elif(abs(test[0])<abs(test[3]['5%'])):
	        print('Passed KPSS @ 5%')
	    elif(abs(test[0])<abs(test[3]['2.5%'])):
	        print('Passed KPSS @ 2.5%')
	    elif(abs(test[0])<abs(test[3]['1%'])):
	        print('Passed KPSS @ 1%')
	    else:
	        print('Failed KPSS: Non-stationary (Trend)')
