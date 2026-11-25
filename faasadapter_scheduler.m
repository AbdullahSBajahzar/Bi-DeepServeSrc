function [assign, metrics] = faasadapter_scheduler(comp_demand, ...
    mem_footprint, container_size, cap_cpu, cap_mem, ...
    invocations, latencies, sla_lat, t, warmup_threshold)
% FaaSAdapter main scheduler
% Workflow-aware execution prediction + resource configuration

[N, ~] = size(cap_cpu);
I_func = length(comp_demand);

% Track last invocation time per function
last_inv = zeros(I_func, 1);
for i = 1:I_func
    for k = t-1:-1:max(1, t-warmup_threshold)
        if invocations(i, k) > 0
            last_inv(i) = k;
            break;
        end
    end
end

% Predict execution time for each function on each node
exec_time = zeros(N, I_func);
for i = 1:I_func
    is_warm = faasadapter_warmup(last_inv(i), t, warmup_threshold);
    for n = 1:N
        [cpu_a, mem_a] = faasadapter_resource(comp_demand(i), ...
            mem_footprint(i), cap_cpu(n), cap_mem(n));
        if cpu_a >= comp_demand(i)/1000 && mem_a >= mem_footprint(i)
            exec_time(n, i) = faasadapter_exec_time(comp_demand(i), ...
                mem_footprint(i), container_size(i), cap_cpu(n), ...
                cap_mem(n), is_warm);
        else
            exec_time(n, i) = inf;
        end
    end
end

% Assign each function to the node with minimum predicted execution time
% (no SLA filtering - all functions are assigned)
assign = zeros(I_func, 1);
for i = 1:I_func
    [min_time, best_n] = min(exec_time(:, i));
    if min_time < inf
        assign(i) = best_n;
    end
end

% Compute metrics
lat_total = 0; sla_count = 0; thr_count = 0;
for i = 1:I_func
    if assign(i) > 0
        obs_lat = latencies(i, t);
        lat_total = lat_total + obs_lat;
        if obs_lat > sla_lat(i)
            sla_count = sla_count + 1;
        end
        thr_count = thr_count + 1;
    end
end

if thr_count > 0
    avg_lat = lat_total / thr_count;
else
    avg_lat = 0;
end
sla_rate = 100 * sla_count / max(I_func, 1);

metrics = struct('latency', avg_lat, 'sla_viol', sla_rate, ...
                 'throughput', thr_count);
end