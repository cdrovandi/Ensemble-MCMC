load('data_Lorenz63.mat')

% True parameter values (not on log scale)
thetaA = [10 28 8/3]; % Main parameters
thetaB = [sqrt(2) sqrt(2) sqrt(2)]; % Observation variances

% Values of parameter to evaluate
nreps = 5;
theta1_vals = repmat(1:20, 1, nreps);
nvals = length(theta1_vals);

% ENKF

rng(1);

N = 100; % number of particles
sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;

tic;
ll_estimates_enkf = zeros(1, nvals);
for i = 1:nvals
    i
    thetaA(1) = theta1_vals(i);
    ll_estimates_enkf(i) = EnKF(xinit,Y,T,N,thetaA,thetaB,sim_noise_scale,dt,steps_per_obs);
end
t_enkf = toc;
t_enkf / nvals

% BPF

tic;
N = 100;
ll_estimates_bpf = zeros(1, nvals);
for i = 1:nvals
    i
    thetaA(1) = theta1_vals(i);
    ll_estimates_bpf(i) = BPF(xinit,Y,T,N,thetaA,thetaB,sim_noise_scale,dt,steps_per_obs);
end
t_bpf = toc;
t_bpf / nvals

figure;
scatter(theta1_vals, ll_estimates_enkf, 'rx');
hold on;
scatter(theta1_vals, ll_estimates_bpf, 'bo');
plot([10 10], ylim, 'Color', 'black'); % True parameter value
xlabel('$\theta_1$','interpreter','latex')
ylabel('log likelihood')

set(gcf, 'PaperPosition', [0 0 10 8]);
set(gcf, 'PaperSize', [10 8]);
saveas(gcf, 'loglike', 'pdf');

figure;
toplot = (theta1_vals >= 8 & theta1_vals <= 12);
scatter(theta1_vals(toplot), ll_estimates_enkf(toplot), 'rx');
hold on;
scatter(theta1_vals(toplot), ll_estimates_bpf(toplot), 'bo');
plot([10 10], [-180 -340], 'Color', 'black'); % True parameter value
xlabel('$\theta_1$','interpreter','latex')
ylabel('log likelihood')

set(gcf, 'PaperPosition', [0 0 10 8]);
set(gcf, 'PaperSize', [10 8]);
saveas(gcf, 'loglike_zoom', 'pdf');
