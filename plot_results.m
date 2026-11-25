%% Plot simulation results
clear; clc; close all;

load('simulation_results.mat');

fig = figure('Color','w','Position',[100 100 1000 700]);

% Latency
subplot(2,2,1);
plot(T_valid, ts.latency_bds, 'b-', 'LineWidth', 1.5); hold on;
plot(T_valid, ts.latency_aeasm, 'r--', 'LineWidth', 1.5);
plot(T_valid, ts.latency_faas, 'g:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Latency (ms)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('Average latency'); grid on;

% SLA
subplot(2,2,2);
plot(T_valid, ts.sla_bds, 'b-', 'LineWidth', 1.5); hold on;
plot(T_valid, ts.sla_aeasm, 'r--', 'LineWidth', 1.5);
plot(T_valid, ts.sla_faas, 'g:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('SLA violation (%)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('SLA violation'); grid on;

% Throughput
subplot(2,2,3);
plot(T_valid, ts.thr_bds, 'b-', 'LineWidth', 1.5); hold on;
plot(T_valid, ts.thr_aeasm, 'r--', 'LineWidth', 1.5);
plot(T_valid, ts.thr_faas, 'g:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Throughput (req/sec)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('Throughput'); grid on;

% Cost
subplot(2,2,4);
plot(T_valid, ts.cost_bds, 'b-', 'LineWidth', 1.5); hold on;
plot(T_valid, ts.cost_aeasm, 'r--', 'LineWidth', 1.5);
plot(T_valid, ts.cost_faas, 'g:', 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Cost ($)');
legend('Bi-DeepServe','AEASM','FaaSAdapter','Location','best');
title('Operational cost'); grid on;

print(fig, 'simulation_results.png', '-dpng', '-r600');