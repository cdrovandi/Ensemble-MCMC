load('data_Lorenz96.mat')

% ENKF

rng(1);

N = 5000; % number of particles
iters = 10000;
% Initial values of log parameters - currently I'm using the true values
theta0 = [log(1) log(1) log(8) log(5)];
% Estimated posterior covariance from pilot runs
cov_rw = [ 1.72,    1.99,   -0.07,  -0.02;
           1.99,    8.11,    1.52,   0.26;
          -0.07,    1.52,    2.86,  -0.16;
          -0.02,    0.26,   -0.16,   0.47;];
cov_rw = cov_rw / 1000;
cov_scale = 1; %Scale factor for proposal covariance

sim_noise_scale = sqrt(10);
dt = 0.01;
steps_per_obs=20;

tic;
[theta_samp_EnKF] = eMCMC_Lorenz96(iters,xinit,Y,n,T,theta0,cov_rw,N,cov_scale,sim_noise_scale,dt,steps_per_obs);
toc
multiESS(theta_samp_EnKF)

% acc_rate: 0.1826
% t_enkf: 79097
% ESS: 321

% SAVE RESULTS

save('output_Lorenz96.mat', 'theta_samp_EnKF');

% PLOTS

%% MCMC trace plots
figure;
for i = 1:4
    subaxis(2,2,i);
    plot(theta_samp_EnKF(:,i))
end

%% Pairs plot 
figure;
plotmatrix(exp(theta_samp_EnKF));

%% Marginals
ylims = [15, 6, 1, 4];
parname = {'$\theta_1$', '$\theta_2$', '$\theta_3$', '$\sigma$'};

figure;
subaxis('MT',0.03,'MB',0.08,'ML',0.08,'MR',0.05,'PL',0.01,'PR',0.01,'Pt',0.02,'PB',0.03,'spacing',0.05);
for i = 1:4
    subaxis(2,2,i);
    [fx, x] = ksdensity(exp(theta_samp_EnKF(:,i)));
    plot(x,fx,'LineWidth',2);
    hold on;

    v = exp(theta0(i));
    plot([v v], [0 ylims(i)], 'Color', 'black');

    xlabel(parname(i),'interpreter','latex')
end

set(gcf, 'PaperPosition', [0 0 15 10]);
set(gcf, 'PaperSize', [15 10]);
saveas(gcf, 'marginals96', 'pdf') %Save figure
