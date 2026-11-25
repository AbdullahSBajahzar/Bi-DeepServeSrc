function [assign, metrics] = aeasm_scheduler(U_cpu, U_mem, U_net, U_disk, ...
    node_type, t, cap_cpu, cap_mem, req_cpu, req_mem, ...
    latencies, sla_lat)
% AEASM main scheduler: MAPE-K control loop

% Phase 1: Monitor
state = aeasm_monitor(U_cpu, U_mem, U_net, U_disk, node_type, t);

% Phase 2: Analyze
analysis = aeasm_analyze(state);

% Phase 3: Plan
plan = aeasm_plan(analysis, cap_cpu, cap_mem, req_cpu, req_mem);

% Phase 4: Execute
[lat, sla, thr] = aeasm_execute(plan, latencies, sla_lat, t);

assign = plan.assign;
metrics = struct('latency', lat, 'sla_viol', sla, ...
                 'throughput', thr, 'energy', sum(analysis.energy));
end