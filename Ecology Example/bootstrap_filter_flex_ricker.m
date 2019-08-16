function loglike = bootstrap_filter_flex_ricker(theta,y,N)
%%
% BPF for flex ricker model
% inputs:
%   theta - parameter
%   y - data
%   N - number of particles
% outputs:
%   loglike - estimated log-likelihood
%%
    T = length(y);

    b0 = theta(1); b1 = theta(2); b2 = theta(3); sigma_e = theta(4); sigma_w = theta(5); logn0 = theta(6);
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
        
        logn = logn + b0 + b1*exp(logn) + b2*exp(logn).^2 + sigma_e*randn(N,1);
        
        logw = -0.5*log(2*pi*V) - 0.5*(y(t) - logn).^2./V;
        loglike = loglike - log(N) + logsumexp(logw);
        
        logw = logw - max(logw);
        W = exp(logw);
        W = W/sum(W);
        
    end

end

