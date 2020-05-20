function loglike = EnKF(xinit,y,n,T,N,theta,sigma,sim_noise_scale,dt,steps_per_obs)
 
P = eye(n);
S = sigma^2 * eye(n);
 
loglike = 0;
 
for t = 1:T
     
    if (t == 1)
        xs = zeros(N,n);
        for i = 1:N
            xs(i,:) = simulate_Lorenz96_single(xinit, n, theta, sim_noise_scale, dt, steps_per_obs);
        end
    else
        for i = 1:N
            xs(i,:) = simulate_Lorenz96_single(xs(i,:), n, theta, sim_noise_scale, dt, steps_per_obs);
        end
    end   
     
    mu_hat = mean(xs)';
    Sigma_hat = cov(xs);
    Z = P*Sigma_hat*P'+S;

    % Return likelihood zero if Z singular up to finitie precision
    if (not(all(all(isfinite(Z))))) % nb single all acts on columns
	loglike = -Inf;
        return
    end
    [T, num] = cholcov(Z, 0);
    if (num > 0)
	loglike = -Inf;
        return
    end

    loglike = loglike + log(mvnpdf(y(t,:)', P*mu_hat, Z));
    K = Sigma_hat*P'*inv(Z);
    ys = mvnrnd(xs*P', S);
    for i = 1:N
        xs(i,:) = xs(i,:) + (y(t,:) - ys(i,:)) * K';
    end
  
end
 
end
