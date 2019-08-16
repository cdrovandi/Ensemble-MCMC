function loglike = EnKF_rqmc(xinit,y,T,N,theta,sigma,sim_noise_scale,dt,steps_per_obs)
 
P = eye(3);
S = zeros(3);
rootS = zeros(3);
 
for i=1:3
  S(i,i) = sigma(i)^2;
  rootS(i,i) = sigma(i);
end
 
loglike = 0;
 
for t = 1:T
     
    if (t == 1)
        r = gen_Sobol(ceil(log2(N)), 3*steps_per_obs)';
        r = r(1:N,:);
        q = norminv(r);
        xs = zeros(N,3);
        for i = 1:N
            qs = reshape(q(i,:),steps_per_obs,3);
            [xs(i,1), xs(i,2), xs(i,3)] = simulate_Lorenz63_single_frn(xinit, theta, sim_noise_scale, dt, steps_per_obs, qs);
        end
    else
        r = gen_Sobol(ceil(log2(N)),3 + 3*steps_per_obs)'; 
        r = r(1:N,:);
        q = norminv(r);
	ys = xs*P' + q(:,1:3)*rootS;
        for i = 1:N
            xs(i,:) = xs(i,:) + (y(t-1,:) - ys(i,:)) * K';
            qs = reshape(q(i,4:end),steps_per_obs,3);
            [xs(i,1), xs(i,2), xs(i,3)] = simulate_Lorenz63_single_frn(xs(i,:), theta, sim_noise_scale, dt, steps_per_obs, qs);
        end
    end   

    mu_hat = mean(xs)';
    Sigma_hat = cov(xs);    
    loglike = loglike + log(mvnpdf(y(t,:)', P*mu_hat, P*Sigma_hat*P' + S));
    K = Sigma_hat*P'*inv(P*Sigma_hat*P' + S);
end
 
end
