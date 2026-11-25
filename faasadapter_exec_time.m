function exec_time = faasadapter_exec_time(comp_demand, mem_footprint, ...
    container_size, node_cpu_cap, node_mem_cap, warmup_done)
% FaaSAdapter execution time prediction
% Input : comp_demand     - [I x 1] computational demand (Megacycles)
%         mem_footprint   - [I x 1] memory footprint (MB)
%         container_size  - [I x 1] container image size (MB)
%         node_cpu_cap    - scalar, CPU capacity of node
%         node_mem_cap    - scalar, memory capacity of node
%         warmup_done     - scalar, 1 if warm, 0 if cold
% Output: exec_time       - [I x 1] predicted execution time (ms)

I = length(comp_demand);
exec_time = zeros(I, 1);

for i = 1:I
    % Base execution time: comp_demand / (node_cpu_cap * 1000)
    base_time = comp_demand(i) / (node_cpu_cap * 1000) * 1000;  % ms
    
    % Memory penalty: if mem_footprint > node_mem_cap, slowdown
    mem_ratio = mem_footprint(i) / node_mem_cap;
    if mem_ratio > 1
        mem_penalty = 1 + 0.5 * (mem_ratio - 1);
    else
        mem_penalty = 1.0;
    end
    
    % Cold start penalty: container image size / network bandwidth
    if warmup_done == 0
        cold_start = container_size(i) / 100;   % 100 MB/s network
    else
        cold_start = 0;
    end
    
    exec_time(i) = base_time * mem_penalty + cold_start;
end
end