function [L_total, L_lat, L_thr, L_cost] = amlo_loss(lat_pred, lat_obs, ...
    thr_pred, thr_target, cost_used, cost_budget, alpha, beta, gamma)
% AMLO total loss (Eq. 13): weighted sum of three losses
% Input : lat_pred, lat_obs       - [I x 1] latency vectors
%         thr_pred, thr_target    - [I x 1] throughput vectors
%         cost_used, cost_budget  - [N x 1] cost vectors
%         alpha, beta, gamma      - adaptive weights
% Output: L_total                 - scalar total loss
%         L_lat, L_thr, L_cost    - individual losses

L_lat  = latency_loss(lat_pred, lat_obs);
L_thr  = throughput_loss(thr_pred, thr_target);
L_cost = cost_loss(cost_used, cost_budget);

L_total = alpha * L_lat + beta * L_thr + gamma * L_cost;
end