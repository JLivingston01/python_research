
import pandas as pd
import numpy as np
import datetime as dt
import pystan

fakedata=pd.read_csv("c:/users/jliv/downloads/mmmfakedata.csv")
KPI=pd.read_csv("c:/users/jliv/downloads/mmmKPI.csv")

days = [dt.datetime.strftime(dt.datetime(2018,1,1)+dt.timedelta(i),'%a') for i in range(365) ]
media_vars = ['channel_'+str(i) for i in range(8)]
control_vars = list(set(days))
X_media1=fakedata[media_vars]

array = []
for i in range(5):
    fd=X_media1.copy()
    for j in list(fd.columns.values):
        fd[j]=fd[j].shift(i)
    fd.fillna(0,inplace=True)
    fd=np.array(fd)
    array.append(fd)

array=np.array(array)
array.shape
X_media=array
#X_media=np.array(fakedata[media_vars])
X_ctrl=np.array(fakedata[control_vars])
J = len(fakedata)
X_media = X_media.reshape(365,8,5)
dat = {'X_media':X_media,'X_ctrl':X_ctrl}
dat['J']=J
dat['y']=np.array(list(KPI['0']))
dat['sigma']=20
dat['max_lag']=5
dat['num_ctrl']=len(control_vars)
dat['num_media']=len(media_vars)
dat['lag_vec'] = np.array(list(range(dat['max_lag'])))



dat_code_data = """
functions {
// the Hill function
real Hill(real t, real ec, real slope) {
    return 1 / (1 + (t / ec)^(-slope));
    }
// the adstock transformation with a vector of weights
real Adstock(row_vector t, row_vector weights) {
    return dot_product(t, weights) / sum(weights);
    }
  }

data {
    int<lower=0> J;
    vector[J] y;
    int<lower=0> sigma;
    int<lower=1> max_lag;
    int<lower=1> num_media;
    int<lower=1> num_ctrl;
    row_vector[max_lag] lag_vec;
    row_vector[num_ctrl] X_ctrl[J];
    row_vector[max_lag] X_media[J, num_media];
 }
"""
#
dat_code_parameters =   """
parameters {
    real<lower=0> noise_var;
    vector<lower=0>[num_media] beta_medias;
    vector[num_ctrl] beta_ctrl;
    vector<lower=0,upper=1>[num_media] retain_rate;
    vector<lower=0,upper=max_lag-1>[num_media] delay;
    vector<lower=0,upper=1>[num_media] ec;
    vector<lower=0>[num_media] slope;
 }
"""


dat_code_transformed_parameters =   """
transformed parameters {
    // a vector of the mean response
    real mu[J];
    // the cumulative media effect after adstock
    real cum_effect;
    // the cumulative media effect after adstock, and then Hill transformation
    row_vector[num_media] cum_effects_hill[J];
    row_vector[max_lag] lag_weights;
    for (nn in 1:J) {
      for (media in 1 : num_media) {
        for (lag in 1 : max_lag) {
            lag_weights[lag] <- pow(retain_rate[media], (lag - 1 - delay[media]) ^ 2);
            }
        cum_effect <- Adstock(X_media[nn, media], lag_weights);
        cum_effects_hill[nn, media] <- Hill(cum_effect, ec[media], slope[media]);
        }
        mu[nn] <- dot_product(cum_effects_hill[nn], beta_medias) +
                    dot_product(X_ctrl[nn], beta_ctrl);
    }
  }
"""
model =   """
model {
    retain_rate ~ beta(3,3);
    delay ~ uniform(0, max_lag - 1);
    slope ~ gamma(3, 1);
    ec ~ beta(2,2);
    
    for (media_index in 1 : num_media) {
            beta_medias[media_index] ~ normal(1, 3);
            }
    for (ctrl_index in 1 : num_ctrl) {
            beta_ctrl[ctrl_index] ~ normal(1,3);
            }
    noise_var ~ cauchy(0,sigma);
    y ~ normal(mu, noise_var);
  }"""

    
model_code=  dat_code_data+dat_code_parameters+dat_code_transformed_parameters+model
    
    
print("""
      
Compiling the model: 
    
    """)
m = pystan.StanModel(model_code=model_code)
print("""
      
Fitting the model: 
    
    """)
    
#def sampling(self, data=None, pars=None, chains=4, iter=2000,
#                 warmup=None, thin=1, seed=None, init='random',
#                 sample_file=None, diagnostic_file=None, verbose=False,
#                 algorithm=None, control=None, n_jobs=-1, **kwargs)
fit = m.sampling(data=dat, iter=400, chains=4,verbose=True)

fit.summary()

summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])

df.columns.values
rep1 = df[(~df.index.str.contains('cum_effects_hill'))&(~df.index.str.contains('mu'))]
#
diagnose=pystan.check_hmc_diagnostics(fit)

coefs=pd.read_csv("c:/users/jliv/downloads/mmmcoefs.csv")
fit