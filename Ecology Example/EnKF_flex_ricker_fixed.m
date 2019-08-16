function loglike = EnKF_flex_ricker_fixed(theta,y,N,r)
%%
% EnKF for flex ricker model with fixed random numbers
% inputs:
%   theta - parameter
%   y - data
%   N - number of particles
%   r - fixed random numbers
% outputs:
%   loglike - estimated log-likelihood
%%
    T = length(y);

    b0 = theta(1); b1 = theta(2); b2 = theta(3); sigma_e = theta(4); sigma_w = theta(5); logn0 = theta(6);
    R = sigma_w^2;
    H = 1;

    loglike = 0;
    logn = logn0*ones(N,1);
    pos = 1;
    for t = 1:T
        
        % forecase ensemble
        logn = logn + b0 + b1*exp(logn) + b2*exp(logn).^2 + sigma_e*r(:,pos);
        pos = pos+1;
        
        mu_tilde = mean(logn);
        Sigma_tilde = var(logn);

        % likelihood update
        loglike = loglike + log(normpdf(y(t), H*mu_tilde, sqrt(H^2*Sigma_tilde + R)));
        
        K = Sigma_tilde*H/(H^2*Sigma_tilde + R);
        
        % shift ensemble
        ys = H*logn + sigma_w*r(:,pos);
        pos = pos+1;
        logn = logn + K*(y(t) - ys);
        
    end

end

