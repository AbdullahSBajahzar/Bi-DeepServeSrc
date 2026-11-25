%% Plot all comparison figures
% MATLAB R2015a compatible
% Requires: final_results.mat

clear; clc; close all;

%% ---------- Load results ----------
load('final_results.mat');   % loads metrics_bds, metrics_aeasm, metrics_faas, T_valid

%% ---------- Figure 1: Time-series comparison (Fig. 18 style) ----------
fig1 = figure('Color','w','Position',[100 100 1000 700]);

% (a) Latency
subplot(2,2,1);
plot(T_valid, metrics_bds(:,1),   'b-',  'LineWidth', 1.5); hold on;
plot(T_valid, metrics_aeasm(:,1), 'r--', 'LineWidth', 1.5);
plot(T_valid, metrics_faas(:,1),  'g:',  'LineWidth', 1.5);
xlabel('Time step'); ylabel('Latency (ms)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('(a) Average latency'); grid on;

% (b) SLA
subplot(2,2,2);
plot(T_valid, metrics_bds(:,3),   'b-',  'LineWidth', 1.5); hold on;
plot(T_valid, metrics_aeasm(:,3), 'r--', 'LineWidth', 1.5);
plot(T_valid, metrics_faas(:,3),  'g:',  'LineWidth', 1.5);
xlabel('Time step'); ylabel('SLA violation (%)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('(b) SLA violation rate'); grid on;

% (c) Throughput
subplot(2,2,3);
plot(T_valid, metrics_bds(:,4),   'b-',  'LineWidth', 1.5); hold on;
plot(T_valid, metrics_aeasm(:,4), 'r--', 'LineWidth', 1.5);
plot(T_valid, metrics_faas(:,4),  'g:',  'LineWidth', 1.5);
xlabel('Time step'); ylabel('Throughput (req/sec)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('(c) Throughput'); grid on;

% (d) Cost
subplot(2,2,4);
plot(T_valid, metrics_bds(:,7),   'b-',  'LineWidth', 1.5); hold on;
plot(T_valid, metrics_aeasm(:,7), 'r--', 'LineWidth', 1.5);
plot(T_valid, metrics_faas(:,7),  'g:',  'LineWidth', 1.5);
xlabel('Time step'); ylabel('Cost ($)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('(d) Operational cost'); grid on;

print(fig1, 'Fig_comparison_timeseries.png', '-dpng', '-r600');
print(fig1, 'Fig_comparison_timeseries.pdf', '-dpdf', '-painters');

%% ---------- Figure 2: Bar chart of average metrics ----------
avg_bds   = mean(metrics_bds, 1);
avg_aeasm = mean(metrics_aeasm, 1);
avg_faas  = mean(metrics_faas, 1);

fig2 = figure('Color','w','Position',[100 100 1200 600]);
bar_data = [avg_bds; avg_aeasm; avg_faas];
hb = bar(bar_data);
set(gca, 'XTickLabel', {'Bi-DeepServe','AEASM','FaaSAdapter'});
ylabel('Normalized value (per metric)');
title('Average metrics comparison');
legend({'AvgLat','p99','SLA','Thr','Energy','Util','Cost','Cold','RMSE','F1'}, ...
    'Location','eastoutside');
grid on;
print(fig2, 'Fig_comparison_bars.png', '-dpng', '-r600');
print(fig2, 'Fig_comparison_bars.pdf', '-dpdf', '-painters');

%% ---------- Figure 3: Ablation-style 6-panel (Fig. 19 style) ----------
fig3 = figure('Color','w','Position',[100 100 1400 900]);

panel_titles = {'(a) Latency','(b) SLA violation','(c) Throughput', ...
    '(d) Energy','(e) Cost','(f) Cold start'};
idx = [1, 3, 4, 5, 7, 8];

for p = 1:6
    subplot(2,3,p);
    vals = [avg_bds(idx(p)), avg_aeasm(idx(p)), avg_faas(idx(p))];
    bar(vals);
    set(gca, 'XTickLabel', {'Bi-DS','AEASM','FaaS'});
    title(panel_titles{p});
    grid on;
end

print(fig3, 'Fig_ablation_style.png', '-dpng', '-r600');
print(fig3, 'Fig_ablation_style.pdf', '-dpdf', '-painters');

%% ---------- Figure 4: Training loss curve ----------
fig4 = figure('Color','w','Position',[100 100 700 400]);
plot(history.loss(1:length(history.loss)), 'b-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Training loss');
title('Bi-DeepServe training convergence');
grid on;
print(fig4, 'Fig_training_loss.png', '-dpng', '-r600');
print(fig4, 'Fig_training_loss.pdf', '-dpdf', '-painters');

fprintf('All figures generated:\n');
fprintf('  Fig_comparison_timeseries.png/pdf\n');
fprintf('  Fig_comparison_bars.png/pdf\n');
fprintf('  Fig_ablation_style.png/pdf\n');
fprintf('  Fig_training_loss.png/pdf\n');