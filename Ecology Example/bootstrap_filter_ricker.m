function loglike = bootstrap_filter_ricker(theta,y,N)
%%
% BPF for ricker model
% inputs:
%   theta - parameter
%   y - data
%   N - number of particles
% outputs:
%   loglike - estimated log-likelihood
%%
    T = length(y);

    b0 = theta(1); b1 = theta(2); sigma_e = theta(3); sigma_w = theta(4); logn0 = theta(5);
    V = sigma_w^2;
    
    logn = logn0*ones(N,1);
    W = ones(N,1)/N;
    loglike = 0;
    for t = 1:T
        if (any(isinf(W)) || any(isnan(W))) % numerically instability can occur for very unusual simulations, and will be rejected anyway
            loglike = -1e6;
            return;
        end
        r = randsample(1:N,N,'true',W);
        logn = logn(r);
        
        logn = logn + b0 + b1*exp(logn) + sigma_e*randn(N,1);

        logw = -0.5*log(2*pi*V) - 0.5*(y(t) - logn).^2./V;
        loglike = loglike - log(N) + logsumexp(logw);
        
        logw = logw - max(logw);
        W = exp(logw);
        W = W/sum(W);
        
    end

end

