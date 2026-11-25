function L_thr = throughput_loss(thr_pred, thr_target)
% Throughput loss (Eq. 15): Log-ratio penalty
% Input : thr_pred   - [I x 1] predicted throughput
%         thr_target - [I x 1] target throughput
% Output: L_thr      - scalar

I = length(thr_pred);
L_thr = 0;
for i = 1:I
    ratio = thr_pred(i) / max(thr_target(i), 1e-8);
    L_thr = L_thr + log(max(ratio, 1e-8));
end
L_thr = (1/I) * L_thr;
end