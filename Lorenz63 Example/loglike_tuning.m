% Try to tune number of particles to get Var(log likelihood) roughly equal to 1.5

load('data_Lorenz63.mat')

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs = 20;

% True parameter values (not on log scale)
thetaA = [10 28 8/3]; % Main parameters
thetaB = [sqrt(2) sqrt(2) sqrt(2)]; % Observation variances

% Numbers of particles to try
nreps = 30;
nparticles_list = 100:100:3000;
n = length(nparticles_list);

% ENKF
rng(1);

var_ll_estimates_enkf = zeros(1, n);
ll_estimates = zeros(1, nreps);
for i = 1:n
    i
    for j = 1:nreps
        ll_estimates(j) = EnKF(xinit, Y, T, nparticles_list(i), thetaA, thetaB, sim_noise_scale, dt, steps_per_obs);
    end
    var_ll_estimates_enkf(i) = var(ll_estimates);
end

% BPF
rng(2);

var_ll_estimates_bpf = zeros(1, n);
ll_estimates = zeros(1, nreps);
for i = 1:n
    i
    for j = 1:nreps
        ll_estimates(j) = BPF(xinit, Y, T, nparticles_list(i), thetaA, thetaB, sim_noise_scale, dt, steps_per_obs);
    end
    var_ll_estimates_bpf(i) = var(ll_estimates);
end

% RQMC ENKF
rng(3);

var_ll_estimates_enkf_rqmc = zeros(1, n);
ll_estimates = zeros(1, nreps);
for i = 1:n
    i
    for j = 1:nreps
        ll_estimates(j) = EnKF_rqmc(xinit, Y, T, nparticles_list(i), thetaA, thetaB, sim_noise_scale, dt, steps_per_obs);
    end
    var_ll_estimates_enkf_rqmc(i) = var(ll_estimates);
end


figure;
plot(nparticles_list, log(var_ll_estimates_enkf));
hold on;
plot(nparticles_list, log(var_ll_estimates_bpf));
plot(nparticles_list, log(var_ll_estimates_enkf_rqmc));
xlabel('particles')
ylabel('log variance of log likelihood')
legend({'enkf','bpf','enkf_rqmc'})
plot(xlim, [log(1.5) log(1.5)], 'Color', 'black');
