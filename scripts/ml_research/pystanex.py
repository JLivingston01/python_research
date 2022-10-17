# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:08:29 2019

@author: jliv
"""

import pystan
#import os
#os.environ["CC"] = "mingw32" 

schools_code = """
data {
    int<lower=0> J; // number of schools
    vector[J] y; // estimated treatment effects
    vector<lower=0>[J] sigma; // s.e. of effect estimates
}
parameters {
    real mu;
    real<lower=0> tau;
    vector[J] eta;
}
transformed parameters {
    vector[J] theta;
    theta = mu + tau * eta;
}
model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
}
"""

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

sm = pystan.StanModel(model_code=schools_code)
fit = sm.sampling(data=schools_dat, iter=1000, chains=4)
#import distutils
#print(distutils.__file__)

import pystan
print(pystan)


from Cython.Build.Inline import _get_build_extension
_get_build_extension().compiler

platform.platform()