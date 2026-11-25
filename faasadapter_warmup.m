function is_warm = faasadapter_warmup(last_invocation, current_time, ...
    warmup_threshold)
% FaaSAdapter warm-up management
% A function is warm if invoked within warmup_threshold seconds
% Input : last_invocation  - last invocation time (or 0 if never)
%         current_time     - current time
%         warmup_threshold - warm-up threshold (seconds)
% Output: is_warm          - 1 if warm, 0 if cold

if last_invocation == 0
    is_warm = 0;
elseif (current_time - last_invocation) <= warmup_threshold
    is_warm = 1;
else
    is_warm = 0;
end
end