function [latency, sla_viol, throughput] = aeasm_execute(plan, ...
    latencies, sla_lat, t)
% MAPE-K Execute phase: compute metrics for the chosen assignment

I_func = length(plan.assign);
latency = 0; sla_viol = 0; throughput = 0;

for i = 1:I_func
    if plan.assign(i) == 0
        continue;
    end
    obs_lat = latencies(i, t);
    latency = latency + obs_lat;
    if obs_lat > sla_lat(i)
        sla_viol = sla_viol + 1;
    end
    throughput = throughput + 1;
end

if throughput > 0
    latency = latency / throughput;
end
sla_viol = 100 * sla_viol / max(I_func, 1);
end