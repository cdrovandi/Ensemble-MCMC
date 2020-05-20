load('data_Lorenz63.mat')

% Try varying number of particles used in eMCMC from our tuned choice of 500

iters = 10000;
theta0 = [log(10) log(28) log(8/3) log(2)/2 log(2)/2 log(2)/2];
cov_rw = [ 0.00322,  0.00006, -0.00006, -0.00325, -0.00268,  0.00243;
           0.00006,  0.00018, -0.00001,  0.00017,  0.00012, -0.00081;
          -0.00006, -0.00001,  0.00113,  0.00177,  0.00052, -0.00231;
          -0.00325,  0.00017,  0.00177,  0.07065,  0.00977, -0.02446;
          -0.00268,  0.00012,  0.00052,  0.00977,  0.03073, -0.01495;
           0.00243, -0.00081, -0.00231, -0.02446, -0.01495,  0.07936 ];
cov_scale = 1; %Scale factor for proposal covariance
rng(1);
sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;
m = 24; %Maximum number of particles to use (in multiples of 25)
time_vals = zeros(1, m);
ess_vals = zeros(1, m);

for i = 1:m
    i
    N = 25*i; % number of particles
    tic;
    [theta_samp_EnKF] = bayes_mcmc_EnKF(iters,xinit,Y,T,theta0,cov_rw,N,cov_scale,sim_noise_scale,dt,steps_per_obs);
    time_vals(i) = toc;
    ess_vals(i) = multiESS(theta_samp_EnKF);
    % Save output to analyse offline
    save(strcat('nparticles_comparison', num2str(i), '.mat'), 'theta_samp_EnKF');
end

figure;
scatter(25*(1:m), ess_vals);
figure;
scatter(25*(1:m), ess_vals ./ time_vals);
