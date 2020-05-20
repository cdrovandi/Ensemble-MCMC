load('data_Lorenz63.mat')

rng(1);

% Number of particles
Ne = 500; 
Np = 2500;
iters = 10000;
% Initial values of log parameters - currently I'm using the true values
theta0 = [log(10) log(28) log(8/3) 0 0 0];
% Proposal covariances taken from pilot runs
cov_rw_e = [ 0.00322,  0.00006, -0.00006, -0.00325, -0.00268,  0.00243;
             0.00006,  0.00018, -0.00001,  0.00017,  0.00012, -0.00081;
            -0.00006, -0.00001,  0.00113,  0.00177,  0.00052, -0.00231;
            -0.00325,  0.00017,  0.00177,  0.07065,  0.00977, -0.02446;
            -0.00268,  0.00012,  0.00052,  0.00977,  0.03073, -0.01495;
             0.00243, -0.00081, -0.00231, -0.02446, -0.01495,  0.07936 ];
cov_rw_p = [ 0.00280, -0.00002,  0.00057, -0.00077, -0.00235,  0.00117;
            -0.00002,  0.00017,  0.00001,  0.00032,  0.00028, -0.00116;
             0.00057,  0.00001,  0.00144,  0.00035, -0.00016, -0.00364;
            -0.00077,  0.00032,  0.00035,  0.05094,  0.00630, -0.01255;
            -0.00235,  0.00028, -0.00016,  0.00630,  0.03658, -0.01988;
             0.00117, -0.00116, -0.00364, -0.01255, -0.01988,  0.12289 ];
cov_scale = 1; %Scale factor for proposal covariance

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;
mse_emcmc = zeros(1,10);
mse_pmcmc = zeros(1,10);

for i = 1:10
  obs_noise_scale = i * 0.2;
  Y = X + randn([T,3]) * obs_noise_scale;
  [theta_samp_emcmc] = bayes_mcmc_EnKF(iters,xinit,Y,T,theta0,cov_rw_e,Ne,cov_scale,sim_noise_scale,dt,steps_per_obs);
  [theta_samp_pmcmc] = bayes_mcmc_BPF(iters,xinit,Y,T,theta0,cov_rw_p,Np,cov_scale,sim_noise_scale,dt,steps_per_obs);
  for j=1:6
    err = exp(theta_samp_emcmc(:,j)) - exp(theta0(j));
    mse_emcmc(i) = mse_emcmc(i) + sum(err.^2);
    err = exp(theta_samp_pmcmc(:,j)) - exp(theta0(j));
    mse_pmcmc(i) = mse_pmcmc(i) + sum(err.^2);
  end
  mse_emcmc(i) = mse_emcmc(i) / iters;
  mse_pmcmc(i) = mse_pmcmc(i) / iters;
end

save('obs_scale_comparison.mat', 'mse_emcmc', 'mse_pmcmc');

% mse_emcmc
%1.8186    1.5918    1.4846    1.3590    0.6462    1.7475    2.5948    2.5290    6.3656    7.3614
% mse_pmcmc
%1.7040    1.2581    1.2646    1.2772    0.5425    1.5279    2.2161    2.4184    7.1086    7.2157
% mse_emcmc ./ mse_pmcmc
%1.0673    1.2652    1.1739    1.0640    1.1913    1.1437    1.1709    1.0457    0.8955    1.0202
