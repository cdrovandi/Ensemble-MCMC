# Code for Bayesian inference of a discretised Lorenz63 SDE.

## Results
* `output_Lorenz96.mat` - saved MCMC output

## Our code

* `data_Lorenz96.mat` - saved simulated data
* `eMCMC_Lorenz96.m` - eMCMC function
* `EnKF_Lorenz96.m` - ensemble Kalman filter function
* `loglike_tuning_Lorenz96.m` - runs an analysis to tune number of particles to use
* `run_experiments_Lorenz96.m` - script to do inference and plot results
* `simulate_Lorenz96_data.m` - script to generate simulated data
* `simulate_Lorenz96_single.m` - forward simulate once from model

## Other code

* `multiESS.m` - calculates ESS of a Markov chain
* `subaxis.m`, `subaxis_license.txt`, `parseArgs.m` - files for a plotting utility
