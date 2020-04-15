function loglike = BPF(xinit,y,n,T,N,theta,sigma,sim_noise_scale,dt,steps_per_obs)

H = eye(n);
R = sigma^2 * eye(n);

loglike = 0;

X = repmat(xinit,N,1);
W = ones(N,1)/N;
logW = zeros(N, 1);

for t = 1:T
    
    r = randsample(1:N,N,'true',W);
    X = X(r,:);
    
    for i = 1:N
        X(i,:) = simulate_Lorenz63_single(X(i,:), theta, sim_noise_scale, dt, steps_per_obs);
        logW(i) = logmvnpdf(y(t,:),X(i,:),R); %Working with log pdfs avoids problems when all pdfs v close to zero
    end
    
    loglike = loglike + log(mean(exp(logW)));
    logW = logW - logsumexp(logW); %Normalise so sum(exp(logW)) = 1
    W = exp(logW);
    W = W/sum(W); %Normalise again in case exponentiating introduced rounding errors
end


end
