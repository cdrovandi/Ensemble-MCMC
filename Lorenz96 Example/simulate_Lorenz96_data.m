T = 30; % number of observations
n = 50; % state dimension
xinit = zeros(1, n);
theta0 = [1, 1, 8]; %Standard values from literature
sim_noise_scale = sqrt(10);
obs_noise_scale = 5;
dt = 0.01;
steps_per_obs = 20;


% Just simulate data at observation times (more efficient)

rng(1);

X = zeros(T,n); % Matrix of true state values at observation times

x = xinit;
for t = 1:T
  x = simulate_Lorenz96_single(x, n, theta0, sim_noise_scale, dt, steps_per_obs);
  X(t,:) = x;
end

Y = X + randn([T,n]) * obs_noise_scale;


% Simulate true state at finer grid (less efficient but useful for plot)

rng(1);

M = T*steps_per_obs;
Xfull = zeros(M,n); % Matrix of true state values at observation times

x = xinit;
for t = 1:M
  x = simulate_Lorenz96_single(x, n, theta0, sim_noise_scale, dt, 1);
  Xfull(t,:) = x;
end

save('data_Lorenz96.mat', 'T', 'n', 'xinit', 'dt', 'steps_per_obs', 'X', 'Y', 'Xfull');

figure;
plot(0.01:0.01:6, Xfull(:,1), 'r');
hold on;
scatter(0.2:0.2:6, Y(:,1), 'ro', 'filled');
plot(0.01:0.01:6, Xfull(:,2), ':b');
scatter(0.2:0.2:6, Y(:,2), 'bd', 'filled');
plot(0.01:0.01:6, Xfull(:,3), '--g');
scatter(0.2:0.2:6, Y(:,3), 'gs', 'filled');
xlabel('t')
