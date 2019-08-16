%% DATA

load('data_Lorenz63.mat');
figure;
plot(0.01:0.01:6, Xfull(:,1), 'r');
hold on;
scatter(0.2:0.2:6, Y(:,1), 'ro', 'filled');
plot(0.01:0.01:6, Xfull(:,2), ':b');
scatter(0.2:0.2:6, Y(:,2), 'bd', 'filled');
plot(0.01:0.01:6, Xfull(:,3), '--g');
scatter(0.2:0.2:6, Y(:,3), 'gs', 'filled');
xlabel('t')

set(gcf, 'PaperPosition', [0 0 12 8]);
set(gcf, 'PaperSize', [12 8]);
saveas(gcf, 'data_Lorenz63', 'pdf');

%% MCMC OUTPUT

load('output_Lorenz63.mat');

%% Univariate / bivariate marginal plots

figure;
plotmatrix(exp(theta_samp_BPF));

figure;
plotmatrix(exp(theta_samp_EnKF));

figure;
plotmatrix(exp(theta_samp_EnKF_rqmc));

figure;
plotmatrix(exp(theta_samp_EnKF_correlated));

figure;
plotmatrix(exp(theta_samp_pEnKF));

%% Comparison of univariate density estimates

figure;
subaxis('MT',0.03,'MB',0.08,'ML',0.08,'MR',0.05,'PL',0.01,'PR',0.01,'Pt',0.02,'PB',0.03,'spacing',0.05);
theta0 = [log(10) log(28) log(8/3) log(2)/2 log(2)/2 log(2)/2]; %True parameter values
parname = {'$\theta_1$', '$\theta_2$', '$\theta_3$', '$\sigma_1$', '$\sigma_2$', '$\sigma_3$'};
bandwidths = [0.13, 0.1, 0.023, 0.05, 0.07, 0.1]; % Length of support (judged by eye) / 30
ylims = [1, 1.5, 6, 2, 2, 1.5];

for i = 1:6
    subaxis(3,2,i);
    [fx, x] = ksdensity(exp(theta_samp_BPF(:,i)), 'Bandwidth', bandwidths(i));
    plot(x,fx,'b','LineWidth',2);
    hold on
    
    [fx, x] = ksdensity(exp(theta_samp_EnKF(:,i)), 'Bandwidth', bandwidths(i));
    plot(x,fx,'--r','LineWidth',2);

    [fx, x] = ksdensity(exp(theta_samp_EnKF_rqmc(:,i)), 'Bandwidth', bandwidths(i));
    plot(x,fx,':m','LineWidth',2);

    [fx, x] = ksdensity(exp(theta_samp_EnKF_correlated(:,i)), 'Bandwidth', bandwidths(i));
    plot(x,fx,'-.g','LineWidth',2);

    %%[fx, x] = ksdensity(exp(theta_samp_pEnKF(:,i)), 'Bandwidth', bandwidths(i));
    %%plot(x,fx,':g','LineWidth',2);

    v = exp(theta0(i));
    plot([v v], [0 ylims(i)], 'Color', 'black');

    xlabel(parname(i),'interpreter','latex')
end

set(gcf, 'PaperPosition', [0 0 15 15]);
set(gcf, 'PaperSize', [15 15]);
saveas(gcf, 'marginals', 'pdf') %Save figure
