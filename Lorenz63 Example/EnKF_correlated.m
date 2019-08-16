function loglike = EnKF(xinit,y,T,N,theta,sigma,sim_noise_scale,dt,steps_per_obs,rand_vec)
 
P = eye(3);
S = zeros(3);
rootS = zeros(3);
 
for i=1:3
  S(i,i) = sigma(i)^2;
  rootS(i,i) = sigma(i);
end
 
loglike = 0;
pos = 1; % Counter through rand_vec 

for t = 1:T

    if (t == 1)
        xs = zeros(N,3);
        for i = 1:N
            qs = rand_vec(pos:(pos-1+steps_per_obs*3));
            qs = reshape(qs,steps_per_obs,3);
            pos = pos + steps_per_obs*3;
            [xs(i,1), xs(i,2), xs(i,3)] = simulate_Lorenz63_single_frn(xinit, theta, sim_noise_scale, dt, steps_per_obs, qs);
        end
    else
        for i = 1:N
            qs = rand_vec(pos:(pos-1+steps_per_obs*3));
            qs = reshape(qs,steps_per_obs,3);
            pos = pos + steps_per_obs*3;
            [xs(i,1), xs(i,2), xs(i,3)] = simulate_Lorenz63_single_frn(xs(i,:), theta, sim_noise_scale, dt, steps_per_obs, qs);
        end
    end
     
    mu_hat = mean(xs)';
    Sigma_hat = cov(xs);    
    loglike = loglike + log(mvnpdf(y(t,:)', P*mu_hat, P*Sigma_hat*P' + S));
    K = Sigma_hat*P'*inv(P*Sigma_hat*P' + S);
    qs = rand_vec(pos:(pos-1+N*3));
    qs = reshape(qs,N,3);
    pos = pos + N*3;
    ys = xs*P' + qs*rootS;
    for i = 1:N
        xs(i,:) = xs(i,:) + (y(t,:) - ys(i,:)) * K';
    end
  
end
