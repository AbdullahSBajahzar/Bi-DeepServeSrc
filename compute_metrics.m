function metrics = compute_metrics(assign, latencies, sla_lat, t, ...
    U_cpu, U_mem, U_net, U_disk, node_type, cost_budget)
% Compute evaluation metrics for a single time step
% Input : assign     - [I x 1] assigned node per function (0 = unassigned)
%         latencies  - [I x T] observed latency
%         sla_lat    - [I x 1] SLA threshold
%         t          - current time step
%         U_cpu, U_mem, U_net, U_disk - [N x T] utilization
%         node_type  - [N x 1] 1=edge, 2=cloud
%         cost_budget- [N x 1] cost budget per node
% Output: metrics    - struct with latency, sla_viol, throughput, cost, energy

I_func = length(assign);
lat_total = 0; sla_count = 0; thr_count = 0;
cost_total = 0; energy_total = 0;

% Cost rates (Section 3.1.2)
c_cpu = 0.00002; c_mem = 0.00001;
c_net = 0.00012; c_io  = 0.0000004;

for i = 1:I_func
    n = assign(i);
    if n == 0
        continue;
    end
    obs_lat = latencies(i, t);
    lat_total = lat_total + obs_lat;
    if obs_lat > sla_lat(i)
        sla_count = sla_count + 1;
    end
    thr_count = thr_count + 1;
    
    % Cost (Eq. 19)
    cost_total = cost_total + c_cpu*U_cpu(n,t) + c_mem*U_mem(n,t) ...
        + c_net*U_net(n,t) + c_io*U_disk(n,t);
    
    % Energy (Eq. 21)
    if node_type(n) == 1
        P_cpu = 15; P_mem = 5; P_net = 2; P_disk = 1;
    else
        P_cpu = 120; P_mem = 40; P_net = 10; P_disk = 5;
    end
    energy_total = energy_total + P_cpu*U_cpu(n,t) + P_mem*U_mem(n,t) ...
        + P_net*U_net(n,t) + P_disk*U_disk(n,t);
end

if thr_count > 0
    avg_lat = lat_total / thr_count;
else
    avg_lat = 0;
end

metrics = struct();
metrics.latency    = avg_lat;
metrics.sla_viol   = 100 * sla_count / max(I_func, 1);
metrics.throughput = thr_count;
metrics.cost       = cost_total;
metrics.energy     = energy_total;
end