function loglike = EnKF_flex_ricker_rqmc(theta,y,N)
%%
% EnKF for flex ricker model with rqmc extension
% inputs:
%   theta - parameter
%   y - data
%   N - number of particles
% outputs:
%   loglike - estimated log-likelihood
%%
    T = length(y);

    b0 = theta(1); b1 = theta(2); b2 = theta(3); sigma_e = theta(4); sigma_w = theta(5); logn0 = theta(6);
    R = sigma_w^2;
    H = 1;

    loglike = 0;
    for t = 1:T
        
        if (t == 1)
            logn = logn0*ones(N,1);
            % forecase ensemble
            logn = logn + b0 + b1*exp(logn) + b2*exp(logn).^2 + sigma_e*randn(N,1);
        else
            % generate randomised QMC numbers
            % here of dimension two (one for simulating from gaussian approximation of one dimensional filtering distribution and one for simulating from the transition density)
            r = gen_Sobol(ceil(log2(N)),2)'; 
            r = r(1:N,:);
            
            % shift ensemble
            ys = H*logn + sigma_w*r(:,1);
            logn = logn + K*(y(t-1) - ys);
            
            % forecase ensemble
            logn = norminv(r(:,2), logn + b0 + b1*exp(logn) + b2*exp(logn).^2, sigma_e);
        end
        
        mu_tilde = mean(logn);
        Sigma_tilde = var(logn);

        % likelihood update
        loglike = loglike + log(normpdf(y(t), H*mu_tilde, sqrt(H^2*Sigma_tilde + R)));
        
        K = Sigma_tilde*H/(H^2*Sigma_tilde + R);
        
    end

end

