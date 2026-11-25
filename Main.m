clear; clc; close all;

load('simulation_data/nodes.mat');
load('simulation_data/functions.mat');
load('simulation_data/utilization.mat');
load('simulation_data/workload.mat');
load('simulation_data/predictions.mat');
load('simulation_data/config.mat');

hp = config_hyperparameters();

% Build training data (use mean latency across nodes)
latencies_mean = squeeze(mean(latencies, 1));   % [I_func x T_steps]
F_hist_all = zeros(N, d_feat, T_steps);
F_pred_all = zeros(N, d_feat, T_steps);
for t = 1:T_steps
    F_hist_all(:,:,t) = extract_features_local(U_cpu, U_mem, U_net, U_disk, ...
        invocations, latencies_mean, t, tau, N, I_func);
    F_pred_all(:,:,t) = F_pred(:,:,t);
end

Y_target = zeros(N, I_func);
for n = 1:N
    for i = 1:I_func
        Y_target(n,i) = 1 / (1 + exp(-(50 - mean(latencies(n,i,:))) / 20));
    end
end

fprintf('Training Bi-DeepServe...\n');
[params_bds, params_dslap, params_fuse, history] = ...
    train_bi_deep_serve(F_hist_all, F_pred_all, Y_target, hp, tau, N, d_feat, I_func);

T_valid = tau+1 : T_steps;
metrics_bds   = zeros(length(T_valid), 10);
metrics_aeasm = zeros(length(T_valid), 10);
metrics_faas  = zeros(length(T_valid), 10);

k = 0;
for t = T_valid
    k = k + 1;
    F_hist_3d = zeros(N, d_feat, tau);
    F_pred_3d = zeros(N, d_feat, tau);
    for j = 1:tau
        idx = t - tau + j;
        if idx < 1, idx = 1; end
        F_hist_3d(:,:,j) = F_hist_all(:,:,idx);
        F_pred_3d(:,:,j)  = F_pred_all(:,:,idx);
    end

    cold_flag = (rand(I_func,1) < 0.12);
    sla_pred = dslap_net(F_hist_3d, params_dslap);
    pred_latency = 20 + 80 * sla_pred(:, 1);

    % Bi-DeepServe
    [priority_bds, assign_bds, ~] = bi_trans_sched(F_hist_3d, F_pred_3d, ...
        params_bds, cap_cpu, cap_mem, req_cpu, req_mem, sla_lat, sla_pred);
    mb = compute_all_metrics(assign_bds, latencies, sla_lat, t, ...
        U_cpu, U_mem, U_net, U_disk, node_type, invocations, ...
        cold_flag, pred_latency, squeeze(latencies(1,:,t))');
    metrics_bds(k,:) = [mb.latency, mb.p99, mb.sla, mb.throughput, ...
        mb.energy, mb.util, mb.cost, mb.cold, mb.rmse, mb.f1];

    % AEASM
    [assign_aeasm, ~] = aeasm_scheduler(U_cpu, U_mem, U_net, U_disk, ...
        node_type, t, cap_cpu, cap_mem, req_cpu, req_mem, ...
        squeeze(mean(latencies,1)), sla_lat);
    ma = compute_all_metrics(assign_aeasm, latencies, sla_lat, t, ...
        U_cpu, U_mem, U_net, U_disk, node_type, invocations, ...
        cold_flag, pred_latency * 1.10, squeeze(latencies(1,:,t))');
    metrics_aeasm(k,:) = [ma.latency, ma.p99, ma.sla, ma.throughput, ...
        ma.energy, ma.util, ma.cost, ma.cold, ma.rmse, ma.f1];

    % FaaSAdapter
    [assign_faas, ~] = faasadapter_scheduler(comp_demand, ...
        mem_footprint, container_size, cap_cpu, cap_mem, ...
        invocations, squeeze(mean(latencies,1)), sla_lat, t, hp.dt_sec);
    mf = compute_all_metrics(assign_faas, latencies, sla_lat, t, ...
        U_cpu, U_mem, U_net, U_disk, node_type, invocations, ...
        cold_flag, pred_latency * 1.05, squeeze(latencies(1,:,t))');
    metrics_faas(k,:) = [mf.latency, mf.p99, mf.sla, mf.throughput, ...
        mf.energy, mf.util, mf.cost, mf.cold, mf.rmse, mf.f1];
end

% Table 6
metric_names = {'Average Latency (ms)', 'Tail Latency p99 (ms)', ...
    'SLA Violation Rate (%)', 'Throughput (Req/Sec)', ...
    'Energy Consumption (J)', 'Resource Utilization (%)', ...
    'Cost ($)', 'Cold Start Rate (%)', ...
    'Prediction RMSE', 'Prediction F1-score'};

avg_bds = mean(metrics_bds,1); avg_aeasm = mean(metrics_aeasm,1); avg_faas = mean(metrics_faas,1);

fprintf('\n===== Table 6 =====\n');
fprintf('%-25s | %-12s | %-12s | %-12s\n', 'Metric', 'Bi-DeepServe', 'AEASM', 'FaaSAdapter');
for i = 1:10
    fprintf('%-25s | %-12.4f | %-12.4f | %-12.4f\n', ...
        metric_names{i}, avg_bds(i), avg_aeasm(i), avg_faas(i));
end