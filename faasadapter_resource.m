function [cpu_alloc, mem_alloc] = faasadapter_resource(comp_demand, ...
    mem_footprint, node_cpu_cap, node_mem_cap)
% FaaSAdapter resource allocation
% Allocates resources proportional to demand, capped by node capacity

I = length(comp_demand);
cpu_alloc = zeros(I, 1);
mem_alloc = zeros(I, 1);

for i = 1:I
    cpu_alloc(i) = min(comp_demand(i) / 1000, node_cpu_cap);
    mem_alloc(i) = min(mem_footprint(i), node_mem_cap);
end
end