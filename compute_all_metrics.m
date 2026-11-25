function m = compute_all_metrics(assign, latencies, sla_lat, t, ...
    U_cpu, U_mem, U_net, U_disk, node_type, invocations, ...
    cold_start_flag, pred_latency, true_latency)
% Compute all 10 metrics using node-dependent latency

I_func = length(assign);
valid = find(assign > 0);
n_valid = length(valid);

if n_valid == 0
    m = struct('latency',0,'p99',0,'sla',0,'throughput',0, ...
        'energy',0,'util',0,'cost',0,'cold',0,'rmse',0,'f1',0);
    return;
end

% Node-dependent latency: latencies(node, function, time)
lat_vals = zeros(n_valid, 1);
for k = 1:n_valid
    i = valid(k);
    n = assign(i);
    lat_vals(k) = latencies(n, i, t);
end
avg_lat = mean(lat_vals);
p99_lat = prctile(lat_vals, 99);

sla_count = sum(lat_vals > sla_lat(valid));
sla_rate = 100 * sla_count / max(n_valid, 1);

thr = n_valid;

energy = 0;
for k = 1:n_valid
    n = assign(valid(k));
    if node_type(n) == 1, P = [15 5 2 1];
    else, P = [120 40 10 5]; end
    energy = energy + P(1)*U_cpu(n,t) + P(2)*U_mem(n,t) ...
        + P(3)*U_net(n,t) + P(4)*U_disk(n,t);
end

util_sum = 0;
for k = 1:n_valid
    n = assign(valid(k));
    util_sum = util_sum + 100 * (U_cpu(n,t) + U_mem(n,t)) / 2;
end
util = util_sum / max(n_valid, 1);

c_cpu = 0.00002; c_mem = 0.00001; c_net = 0.00012; c_io = 0.0000004;
cost = 0;
for k = 1:n_valid
    n = assign(valid(k));
    cost = cost + c_cpu*U_cpu(n,t) + c_mem*U_mem(n,t) ...
        + c_net*U_net(n,t) + c_io*U_disk(n,t);
end
cost = cost + 0.001 * sum(cold_start_flag(valid));

cold_count = sum(cold_start_flag(valid));
cold_rate = 100 * cold_count / max(n_valid, 1);

if ~isempty(pred_latency) && ~isempty(true_latency)
    diff = pred_latency(valid) - true_latency(valid);
    rmse = sqrt(mean(diff.^2));
else
    rmse = 0;
end

if ~isempty(pred_latency) && ~isempty(true_latency)
    pred_viol = pred_latency(valid) > sla_lat(valid);
    true_viol = true_latency(valid) > sla_lat(valid);
    tp = sum(pred_viol & true_viol);
    fp = sum(pred_viol & ~true_viol);
    fn = sum(~pred_viol & true_viol);
    if (tp + fp) > 0 && (tp + fn) > 0
        prec = tp / (tp + fp);
        rec  = tp / (tp + fn);
        f1 = 2 * prec * rec / (prec + rec + 1e-8);
    else
        f1 = 0;
    end
else
    f1 = 0;
end

m = struct('latency', avg_lat, 'p99', p99_lat, 'sla', sla_rate, ...
    'throughput', thr, 'energy', energy, 'util', util, ...
    'cost', cost, 'cold', cold_rate, 'rmse', rmse, 'f1', f1);
end