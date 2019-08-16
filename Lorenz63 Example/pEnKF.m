%Algorithm 4 of Katzfuss et al (https://arxiv.org/abs/1704.06988)
function [thetas] = pEnKF(xinit,y,T,h,M,N,sim_noise_scale,dt,steps_per_obs)
%Parameters:
%
% xinit - initial state variables
% y - matrix of observations (T rows, 3 columns)
% T - number of observations
% h - scale factor for artificial parameter noise
% M - number of theta particles
% N - number of x particles per ensemble
% sim_noise_scale - scale of noise added at each simulation step
% dt - time step used in simulator
% steps_per_obs - how many simulator time steps to perform between observations

H = eye(3); %Observation matrix

%Initialise various storage variables
prior_mean = 10.0; % Prior mean for all parameters (on log scale)
imp_mean = 1.0; % Importance density mean for all parameters (on log scale)
thetas = exprnd(imp_mean, M, 6); % Sample from importance density
mu_hat = zeros(M, 3);
Sigma_hat = zeros(M, 3, 3);
%weights = zeros(M,1);
weights = prod(exppdf(thetas, prior_mean), 2) ./ prod(exppdf(thetas, imp_mean), 2); % Importance weights
delta = 1E-6; %A small value
max_x = 1E6; %Simulated x values above this result in an ensemble getting zero weight

for t = 1:T
    t
    %Update thetas
    theta_var = cov(thetas) + delta*diag(3);
    theta_var = 0.5 * (theta_var + theta_var'); %Ensure positive definite (might not be due to rounding errors)
    thetas = thetas + mvnrnd(zeros(1,6), h^2*theta_var, M);
    %Loop over x ensembles
    for i = 1:M
        %Propagate ensemble
        if (t == 1)
            xs = zeros(N,3);
            for j = 1:N
                [xs(j,1), xs(j,2), xs(j,3)] = simulate_Lorenz63_single(xinit, exp(thetas(i,1:3)), sim_noise_scale, dt, steps_per_obs);
            end
        else
            for j = 1:N
                S = squeeze(Sigma_hat(i,:,:));
                x_temp = mvnrnd(mu_hat(i,:)', S, 1);
                [xs(j,1), xs(j,2), xs(j,3)] = simulate_Lorenz63_single(x_temp, exp(thetas(i,1:3)), sim_noise_scale, dt, steps_per_obs);
            end
        end
        %Summarise ensemble and calculate likelihood to use as weight
        mu_tilde = mean(xs)';
        Sigma_tilde = cov(xs);
        Sigma_tilde = 0.5 * (Sigma_tilde + Sigma_tilde'); %Ensure positive definite
        R = zeros(3);
        for k=1:3
            R(k,k) = exp(thetas(i,k+3))^2;
        end
        if (any(xs(:)>max_x) || any(isnan(xs(:))) || any(isinf(xs(:))))
            weights(i) = 0.0;
            %Corresponding mu_hat and Sigma_hat entries not updated as won't be used!
        else
            weights(i) = weights(i) * mvnpdf(y(t,:)', H*mu_tilde, H*Sigma_tilde*H' + R);
            K = Sigma_tilde*H'*inv(H*Sigma_tilde*H' + R);
            mu_hat(i,:) = mu_tilde + K*(y(t,:)' - H*mu_tilde);
            Sigma_hat_temp = (eye(3) - K*H)*Sigma_tilde + delta*eye(3);
            Sigma_hat(i,:,:) = 0.5 * (Sigma_hat_temp + Sigma_hat_temp'); % Ensure positive definite
        end
    end %i loop

    %Resample
    ess = sum(weights)^2 / sum(weights.^2)
    indices = randsample(M, M, true, weights); % nb "weights" automatically normalised
    thetas = thetas(indices,:);
    mu_hat = mu_hat(indices,:);
    Sigma_hat = Sigma_hat(indices,:,:);
    weights = ones(M,1);
end %t loop
end %function
