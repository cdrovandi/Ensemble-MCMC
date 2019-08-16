function [theta, loglikes] = pmcmc_ricker(y,theta_init,iters,N,cov_rw)
%%
% pMCMC for Ricker model
% inputs:
%   y - data
%   theta_init - initial parameter value in mcmc chain
%   iters - number of mcmc iterations
%   N - number of particles
%   cov_rw - multivariate normal random walk proposal covariance matrix
% outputs:
%   theta - samples from (approximate) posterior
%   loglikes - the mcmc chain of (estimated) log-likelihoods
%%
theta_curr = theta_init;

loglike_curr =  bootstrap_filter_ricker([theta_curr(1) theta_curr(2) exp(theta_curr(3)) exp(theta_curr(4)) theta_curr(5)],y,N);

logprior_curr = log(normpdf(theta_curr(1),0,1)) + log(normpdf(theta_curr(2),0,1)) + log(exppdf(exp(theta_curr(3)),1)*exp(theta_curr(3))) +  log(exppdf(exp(theta_curr(4)),1)*exp(theta_curr(4)));

theta = zeros(iters,5);
loglikes = zeros(iters,1);

for i = 1:iters
    i
    theta_prop = mvnrnd(theta_curr,cov_rw);
       
    loglike_prop = bootstrap_filter_ricker([theta_prop(1) theta_prop(2) exp(theta_prop(3)) exp(theta_prop(4)) theta_prop(5)],y,N);

    logprior_prop = log(normpdf(theta_prop(1),0,1)) + log(normpdf(theta_prop(2),0,1)) + log(exppdf(exp(theta_prop(3)),1)*exp(theta_prop(3))) +  log(exppdf(exp(theta_prop(4)),1)*exp(theta_prop(4)));
    
    if (rand < exp(loglike_prop - loglike_curr + logprior_prop - logprior_curr))
        fprintf('******* Accepted *********\n');
        theta_curr = theta_prop;
        logprior_curr = logprior_prop;
        loglike_curr = loglike_prop;
    end
    theta(i,:) = theta_curr;
    loglikes(i) = loglike_curr;
    
end

end


