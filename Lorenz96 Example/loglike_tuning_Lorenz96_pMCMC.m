% Try to tune number of particles to get Var(log likelihood) roughly equal to 1.5

load('data_Lorenz96.mat')

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs = 20;

% True parameter values (not on log scale)
thetaA = [1 1 8]; % Main parameters
thetaB = 5; % Observation scale

% Numbers of particles to try
nreps = 30;
nparticles_list = 1000:1000:10000;
ntrials = length(nparticles_list);

rng(1);

var_ll_estimates_bpf = zeros(1, ntrials);
ll_estimates = zeros(1, nreps);
for i = 1:ntrials
    i
    for j = 1:nreps
        ll_estimates(j) = BPF_Lorenz96(xinit, Y, n, T, nparticles_list(i), thetaA, thetaB, sim_noise_scale, dt, steps_per_obs);
    end
    var_ll_estimates_bpf(i) = var(ll_estimates);
end

figure;
plot(nparticles_list, log(var_ll_estimates_bpf));
hold on;
xlabel('particles')
ylabel('log variance of log likelihood')
plot(xlim, [log(1.5) log(1.5)], 'Color', 'black');
