function [theta, loglikes] = pmcmc_EnKF_ricker_correlated_unbiased(y,theta_init,iters,N,cov_rw,sigma_corr)
%%
% eMCMC for Ricker model with correlated and unbiased extensions
% inputs:
%   y - data
%   theta_init - initial parameter value in mcmc chain
%   iters - number of mcmc iterations
%   N - number of particles
%   cov_rw - multivariate normal random walk proposal covariance matrix
%   sigma_corr - tuning parameter for correlated pseudo-marginal
% outputs:
%   theta - samples from (approximate) posterior
%   loglikes - the mcmc chain of (estimated) log-likelihoods
%%
theta_curr = theta_init;
T = length(y);
num_rand = 2*N*T;
rv = randn(1,num_rand);
r = reshape(rv,N,2*T);

loglike_curr =  EnKF_ricker_fixed_unbiased([theta_curr(1) theta_curr(2) exp(theta_curr(3)) exp(theta_curr(4)) theta_curr(5)],y,N,r);

logprior_curr = log(normpdf(theta_curr(1),0,1)) + log(normpdf(theta_curr(2),0,1)) + log(exppdf(exp(theta_curr(3)),1)*exp(theta_curr(3))) +  log(exppdf(exp(theta_curr(4)),1)*exp(theta_curr(4)));

theta = zeros(iters,5);
loglikes = zeros(iters,1);

for i = 1:iters
i    
    theta_prop = mvnrnd(theta_curr,cov_rw);
    rv_prop = sqrt(1 - sigma_corr^2)*rv + sigma_corr*randn(1,num_rand);
    r_prop = reshape(rv_prop,N,2*T);
        
    loglike_prop =   EnKF_ricker_fixed_unbiased([theta_prop(1) theta_prop(2) exp(theta_prop(3)) exp(theta_prop(4)) theta_prop(5)],y,N,r_prop);
    
    logprior_prop = log(normpdf(theta_prop(1),0,1)) + log(normpdf(theta_prop(2),0,1)) + log(exppdf(exp(theta_prop(3)),1)*exp(theta_prop(3))) +  log(exppdf(exp(theta_prop(4)),1)*exp(theta_prop(4)));

    
    if (rand < exp(loglike_prop - loglike_curr + logprior_prop - logprior_curr))
        fprintf('******* Accepted *********\n');
        theta_curr = theta_prop;
        logprior_curr = logprior_prop;
        loglike_curr = loglike_prop;
        rv = rv_prop;
    end
    theta(i,:) = theta_curr;
    loglikes(i) = loglike_curr;
    
end

end
    
    
    