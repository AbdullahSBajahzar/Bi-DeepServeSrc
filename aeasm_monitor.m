function state = aeasm_monitor(U_cpu, U_mem, U_net, U_disk, node_type, t)
% MAPE-K Monitor phase: collect current utilization
% Output: state - struct with current utilization

state = struct();
state.u_cpu  = U_cpu(:, t);
state.u_mem  = U_mem(:, t);
state.u_net  = U_net(:, t);
state.u_disk = U_disk(:, t);
state.node_type = node_type;
end