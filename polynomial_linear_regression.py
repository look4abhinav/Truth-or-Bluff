# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 00:53:54 2018

@author: Abhinav
"""

import numpy as np
import matplotlib.pyplot as mpt
import pandas as pd

pos = input('Enter your Position Level (1-10)')
act_sal = input('Enter your Salary')
error = 0.1

dataset = pd.read_csv('Position_Salaries.csv')
idm = dataset.iloc[:,1:2].values
dm = dataset.iloc[:,2].values

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
idm_poly = poly_reg.fit_transform(idm)
poly_reg.fit(idm_poly,dm)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(idm_poly,dm)

idm_grid = np.arange(min(idm),max(idm),0.01)
idm_grid = idm_grid.reshape(len(idm_grid),1)
mpt.scatter(idm,dm,color='red')
mpt.plot(idm_grid,lin_reg.predict(poly_reg.fit_transform(idm_grid)),color='blue')
mpt.title('Truth or Bluff - Polynomial Regression')
mpt.xlabel('Position Level')
mpt.ylabel('Salary')
mpt.show()

prdt_sal = lin_reg.predict(poly_reg.fit_transform(pos))
err=(prdt_sal-float(act_sal))/prdt_sal
if(err<=error):
    print('Truth')
else:
    print('False')