T = 30; % number of observations
xinit = [0, 0, 0];
theta0 = [10, 28, 8/3]; %Standard values from literature
sim_noise_scale = sqrt(10);
obs_noise_scale = sqrt(2);
dt = 0.01;
steps_per_obs = 20;


% Just simulate data at observation times (more efficient)

rng(1);

X = zeros(T,3); % Matrix of true state values at observation times

x1 = xinit(1); x2 = xinit(2); x3 = xinit(3);
for t = 1:T
  [x1, x2, x3] = simulate_Lorenz63_single([x1, x2, x3], theta0, sim_noise_scale, dt, steps_per_obs);
  X(t,1) = x1; X(t,2) = x2; X(t,3) = x3;
end

Y = X + randn([T,3]) * obs_noise_scale;


% Simulate true state at finer grid (less efficient but useful for plot)

rng(1);

N = T*steps_per_obs;
Xfull = zeros(N,3); % Matrix of true state values at observation times

x1 = xinit(1); x2 = xinit(2); x3 = xinit(3);
for t = 1:N
  [x1, x2, x3] = simulate_Lorenz63_single([x1, x2, x3], theta0, sim_noise_scale, dt, 1);
  Xfull(t,1) = x1; Xfull(t,2) = x2; Xfull(t,3) = x3;
end

save('data_Lorenz63.mat', 'T', 'xinit', 'dt', 'steps_per_obs', 'X', 'Y', 'Xfull');
