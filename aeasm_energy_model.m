function E = aeasm_energy_model(u_cpu, u_mem, u_net, u_disk, node_type)
% AEASM three-level energy model
% Input : u_cpu, u_mem, u_net, u_disk - utilization in [0,1]
%         node_type                   - 1=edge, 2=cloud
% Output: E                           - energy in Joules (per second)

if node_type == 1   % edge
    P_cpu = 15; P_mem = 5; P_net = 2; P_disk = 1;
else                % cloud
    P_cpu = 120; P_mem = 40; P_net = 10; P_disk = 5;
end

% Three-level categorization (idle, active, peak)
if u_cpu < 0.3
    level_factor = 0.7;
elseif u_cpu < 0.7
    level_factor = 1.0;
else
    level_factor = 1.2;
end

E = level_factor * (P_cpu*u_cpu + P_mem*u_mem + P_net*u_net + P_disk*u_disk);
end