# IV curve fitting
voltage was scanned from 0 to 1.75V and from 1.75V to 0. file 'data_firstHalf.csv' and 'data_secHalf.csv' was extracted through 'WebPlotDigitizer'.   
The IV curve (ground truth) is shown as follow:  
<img src="https://github.com/ICscholar/curve_fit/blob/main/raw_fig.png" width="500px">  
The fitted result is shown as follow:  
<img src="https://github.com/ICscholar/curve_fit/blob/main/IV_fitted_curve.png" width="500px">  
There should be 2 curves but 3, since the initial point and end point is linked as well. Not a serious problem.

Curve was fitted based on the equations:  
<img src="https://github.com/ICscholar/curve_fit/blob/main/equations.png" width="900px"> 
