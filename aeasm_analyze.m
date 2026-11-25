function analysis = aeasm_analyze(state)
% MAPE-K Analyze phase: compute energy and identify overloaded nodes
% Output: analysis - struct with energy and overload flags

N = length(state.u_cpu);
energy = zeros(N, 1);
overload = zeros(N, 1);

for n = 1:N
    energy(n) = aeasm_energy_model(state.u_cpu(n), state.u_mem(n), ...
        state.u_net(n), state.u_disk(n), state.node_type(n));
    if state.u_cpu(n) > 0.8 || state.u_mem(n) > 0.8
        overload(n) = 1;
    end
end

analysis = struct();
analysis.energy   = energy;
analysis.overload = overload;
analysis.avg_cpu  = mean(state.u_cpu);
end