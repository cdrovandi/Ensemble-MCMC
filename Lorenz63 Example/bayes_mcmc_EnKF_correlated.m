function [theta_samp] = bayes_mcmc_EnKF_rqmc(iters, xinit, y, T, theta0, cov_rw, N, proposal_scale, sim_noise_scale, dt, steps_per_obs, sigma_corr)
%Parameters:
%
% iters - number of MCMC iterations
% xinit - initial state variables
% y - matrix of observations (T rows, 3 columns)
% T - number of observations
% cov_rw - MCMC proposal covariance matrix (unscaled)
% theta0 - initial log parameter values (3 main parameters followed by 3 observation error parameters)
% N - number of EnKF particles
% proposal_scale - scaling factor for MCMC proposal covariance matrix
% sim_noise_scale - scale of noise added at each simulation step
% dt - time step used in simulator
% steps_per_obs - how many simulator time steps to perform between observations

prior_rate = 0.1; % exponential prior for all parameters with mean 10
theta_samp = zeros(iters,6);
theta = theta0;
acceptances = 0;

logprior_curr = sum(theta - prior_rate*exp(theta));

num_rand = N * T * (6*steps_per_obs + 3);
rand_vec = randn(1, num_rand);

loglike = EnKF_correlated(xinit, y, T, N, exp(theta(1:3)), exp(theta(4:6)), sim_noise_scale, dt, steps_per_obs, rand_vec);

for i = 1:iters
    i
    
    theta_prop = mvnrnd(theta, proposal_scale^2*cov_rw);
    logprior_prop = sum(theta_prop - prior_rate*exp(theta_prop));
    rand_vec_prop = sqrt(1 - sigma_corr^2)*rand_vec + sigma_corr*randn(1, num_rand);
    
    loglike_prop = EnKF_correlated(xinit ,y ,T, N, exp(theta_prop(1:3)), exp(theta_prop(4:6)), sim_noise_scale, dt, steps_per_obs, rand_vec_prop);
    
    log_mh = loglike_prop - loglike + logprior_prop - logprior_curr;
    
    if (rand < exp(log_mh))
        fprintf('*** accepted ***\n');
        acceptances = acceptances + 1;
        theta = theta_prop;
        rand_vec = rand_vec_prop;
        logprior_curr = logprior_prop;
        loglike = loglike_prop;
    end
    
    theta_samp(i,:) = theta;
    
end

acc_rate = acceptances / iters
