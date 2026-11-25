function plan = aeasm_plan(analysis, cap_cpu, cap_mem, req_cpu, req_mem)
% MAPE-K Plan phase: decide on task placement
% Simple heuristic: place on least-loaded non-overloaded node

N = length(analysis.energy);
I_func = length(req_cpu);
assign = zeros(I_func, 1);

for i = 1:I_func
    best_n = 0;
    best_load = inf;
    for n = 1:N
        if analysis.overload(n) == 0 && ...
                req_cpu(i) <= cap_cpu(n) && req_mem(i) <= cap_mem(n)
            if analysis.energy(n) < best_load
                best_load = analysis.energy(n);
                best_n = n;
            end
        end
    end
    assign(i) = best_n;
end

plan = struct();
plan.assign = assign;
end