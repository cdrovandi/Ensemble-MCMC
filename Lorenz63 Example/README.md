# Code for Bayesian inference of a discretised Lorenz63 SDE.

## Results
* `output_Lorenz63.mat` - output from all inference methods
* `nparticles_comparison*.mat` - eMCMC output with different number of particles
* `obs_scale_comparison.mat' - eMCMC and pMCMC under different observation scales

## Our code

* `bayes_mcmc_BPF.m` - pMCMC function
* `bayes_mcmc_EnKF.m` - eMCMC function
* `bayes_mcmc_EnKF_correlated.m` - eMCMC function with correlated noise
* `bayes_mcmc_EnKF_rqmc.m` - eMCMC function with RQMC
* `BPF.m` - bootstrap particle filter function
* `data_Lorenz63.mat` - saved simulated data
* `EnKF.m` - ensemble Kalman filter function
* `EnKF_correlated.m` - ensemble Kalman filter function with correlated noise
* `EnKF_rqmc.m` - ensemble Kalman filter function with RQMC
* `loglike_comparison.m` - runs a comparison of log likelihoods from BPF and EnKF
* `loglike_tuning.m` - runs an analysis to tune number of particles to use
* `nparticles_comparison.m` - runs eMCMC with various numbers of particles
* `nparticles_comparison.R` - examines ESS values using R's coda package
* `obs_scale_comparison.m` - compares MSE from eMCMC and pMCMC under different observation scales
* `pEnKF.m` - particle ensemble Kalman filter function
* `plots.m` - creates plots from saved output
* `run_EnKF_Lorenz63.m` - script to do inference
* `simulate_Lorenz63_data.m` - script to generate simulated data
* `simulate_Lorenz63_single.m` - forward simulate once from model
* `simulate_Lorenz63_single_frn.m` - forward simulate once from model with fixed random numbers

## Other code

* `gen_Sobol.m` - RQMC sequence generation
* `logmvnpdf.m`, `logmvnpdf_license.txt` - files for multivariate normal pdf
* `logsumexp.m` - files for log sum exp function
* `multiESS.m` - calculates ESS of a Markov chain
* `subaxis.m`, `subaxis_license.txt`, `parseArgs.m` - files for a plotting utility
