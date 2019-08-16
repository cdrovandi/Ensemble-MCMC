function loglike = EnKF(xinit,y,T,N,theta,sigma,sim_noise_scale,dt,steps_per_obs)
 
P = eye(3);
S = zeros(3);
 
for i=1:3
  S(i,i) = sigma(i)^2;
end
 
loglike = 0;
 
for t = 1:T
     
    if (t == 1)
        xs = zeros(N,3);
        for i = 1:N
            [xs(i,1), xs(i,2), xs(i,3)] = simulate_Lorenz63_single(xinit, theta, sim_noise_scale, dt, steps_per_obs);
        end
    else
        for i = 1:N
            [xs(i,1), xs(i,2), xs(i,3)] = simulate_Lorenz63_single(xs(i,:), theta, sim_noise_scale, dt, steps_per_obs);
        end
    end   
     
    mu_hat = mean(xs)';
    Sigma_hat = cov(xs);    
    loglike = loglike + log(mvnpdf(y(t,:)', P*mu_hat, P*Sigma_hat*P' + S));
    K = Sigma_hat*P'*inv(P*Sigma_hat*P' + S);
    ys = mvnrnd(xs*P', S);
    for i = 1:N
        xs(i,:) = xs(i,:) + (y(t,:) - ys(i,:)) * K';
    end
  
end
 
end
