load('data_Lorenz63.mat')

% ENKF

rng(1);

N = 500; % number of particles
iters = 10000;
% Initial values of log parameters - currently I'm using the true values
theta0 = [log(10) log(28) log(8/3) log(2)/2 log(2)/2 log(2)/2];
% Proposal covariance taken from pilot runs
cov_rw = [ 0.00322,  0.00006, -0.00006, -0.00325, -0.00268,  0.00243;
           0.00006,  0.00018, -0.00001,  0.00017,  0.00012, -0.00081;
          -0.00006, -0.00001,  0.00113,  0.00177,  0.00052, -0.00231;
          -0.00325,  0.00017,  0.00177,  0.07065,  0.00977, -0.02446;
          -0.00268,  0.00012,  0.00052,  0.00977,  0.03073, -0.01495;
           0.00243, -0.00081, -0.00231, -0.02446, -0.01495,  0.07936 ];

cov_scale = 1; %Scale factor for proposal covariance

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;

tic;
[theta_samp_EnKF] = bayes_mcmc_EnKF(iters,xinit,Y,T,theta0,cov_rw,N,cov_scale,sim_noise_scale,dt,steps_per_obs);
toc
multiESS(theta_samp_EnKF)

% acc_rate: 0.1985
% t_enkf: 689
% ESS: 390.3

% PMCMC

rng(2);

N = 2500;
iters = 10000;
% Proposal covariance taken from pilot runs
cov_rw = [ 0.00280, -0.00002,  0.00057, -0.00077, -0.00235,  0.00117;
          -0.00002,  0.00017,  0.00001,  0.00032,  0.00028, -0.00116;
           0.00057,  0.00001,  0.00144,  0.00035, -0.00016, -0.00364;
          -0.00077,  0.00032,  0.00035,  0.05094,  0.00630, -0.01255;
          -0.00235,  0.00028, -0.00016,  0.00630,  0.03658, -0.01988;
           0.00117, -0.00116, -0.00364, -0.01255, -0.01988,  0.12289 ];
cov_scale = 1; %Scale factor for proposal covariance

tic;
[theta_samp_BPF] = bayes_mcmc_BPF(iters,xinit,Y,T,theta0,cov_rw,N,cov_scale,sim_noise_scale,dt,steps_per_obs);
toc
multiESS(theta_samp_BPF)

% acc_rate: 0.1199
% t_bpf: 10992
% ESS: 197

% pEnKF

rng(3);

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;
h = 0.1;
N = 100; % number of x particles per ensemble
M = 1E4; % number of theta particles

tic;
[theta_samp_pEnKF] = pEnKF(xinit,Y,T,h,M,N,sim_noise_scale,dt,steps_per_obs);
toc
% Smallest ESS: 119 (t=2 iteration)
% t_penkf: 2.5292e+03


% ENKF - rqmc

rng(4);

N = 500; % number of particles
iters = 10000;
% Initial values of log parameters - currently I'm using the true values
theta0 = [log(10) log(28) log(8/3) log(2)/2 log(2)/2 log(2)/2];
% Proposal covariance taken from pilot runs
cov_rw = [ 0.00322,  0.00006, -0.00006, -0.00325, -0.00268,  0.00243;
           0.00006,  0.00018, -0.00001,  0.00017,  0.00012, -0.00081;
          -0.00006, -0.00001,  0.00113,  0.00177,  0.00052, -0.00231;
          -0.00325,  0.00017,  0.00177,  0.07065,  0.00977, -0.02446;
          -0.00268,  0.00012,  0.00052,  0.00977,  0.03073, -0.01495;
           0.00243, -0.00081, -0.00231, -0.02446, -0.01495,  0.07936 ];

cov_scale = 1; %Scale factor for proposal covariance

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;

tic;
[theta_samp_EnKF_rqmc] = bayes_mcmc_EnKF_rqmc(iters,xinit,Y,T,theta0,cov_rw,N,cov_scale,sim_noise_scale,dt,steps_per_obs);
toc
multiESS(theta_samp_EnKF_rqmc)

% acc_rate: 0.1680
% time: 3073
% ESS: 305


% ENKF - correlated

rng(5);

N = 100; % number of particles
iters = 10000;
% Initial values of log parameters - currently I'm using the true values
theta0 = [log(10) log(28) log(8/3) log(2)/2 log(2)/2 log(2)/2];
% Proposal covariance taken from pilot runs
cov_rw = [ 0.00322,  0.00006, -0.00006, -0.00325, -0.00268,  0.00243;
           0.00006,  0.00018, -0.00001,  0.00017,  0.00012, -0.00081;
          -0.00006, -0.00001,  0.00113,  0.00177,  0.00052, -0.00231;
          -0.00325,  0.00017,  0.00177,  0.07065,  0.00977, -0.02446;
          -0.00268,  0.00012,  0.00052,  0.00977,  0.03073, -0.01495;
           0.00243, -0.00081, -0.00231, -0.02446, -0.01495,  0.07936 ];

cov_scale = 1; %Scale factor for proposal covariance
sigma_corr = 0.1;

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;

tic;
[theta_samp_EnKF_correlated] = bayes_mcmc_EnKF_correlated(iters, xinit, Y, T, theta0, cov_rw, N, cov_scale, sim_noise_scale, dt, steps_per_obs, sigma_corr);
toc
multiESS(theta_samp_EnKF_correlated)

% acc_rate: 0.26
% time: 195
% ESS: 417


% SAVE RESULTS

save('output_Lorenz63.mat', 'theta_samp_EnKF', 'theta_samp_pEnKF', 'theta_samp_BPF','theta_samp_EnKF_rqmc', 'theta_samp_EnKF_correlated');


% PLOTS

%% MCMC trace plots
figure;
for i = 1:6
    subaxis(3,2,i);
    plot(theta_samp_BPF(:,i))
end
 
figure;
for i = 1:6
    subaxis(3,2,i);
    plot(theta_samp_EnKF(:,i))
end

figure;
for i = 1:6
    subaxis(3,2,i);
    plot(theta_samp_EnKF_rqmc(:,i))
end

figure;
for i = 1:6
    subaxis(3,2,i);
    plot(theta_samp_EnKF_correlated(:,i))
end
