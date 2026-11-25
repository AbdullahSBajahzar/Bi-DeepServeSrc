function [alpha, beta, gamma] = update_amlo_weights(L_lat, L_thr, L_cost, ...
    alpha, beta, gamma, lr_amlo)
% AMLO adaptive weight update
% The weights are adjusted based on relative magnitudes of losses:
% if a loss is large, its weight is decreased to avoid dominance.
% Input : L_lat, L_thr, L_cost - current losses
%         alpha, beta, gamma    - current weights
%         lr_amlo               - adaptation rate
% Output: updated alpha, beta, gamma

% Normalize losses to [0, 1]
L_sum = L_lat + L_thr + L_cost + 1e-8;
r_lat = L_lat / L_sum;
r_thr = L_thr / L_sum;
r_cost = L_cost / L_sum;

% Update weights: decrease weight for dominant loss
alpha = alpha - lr_amlo * (r_lat - 1/3);
beta  = beta  - lr_amlo * (r_thr - 1/3);
gamma = gamma - lr_amlo * (r_cost - 1/3);

% Clip to [0.1, 0.9] to avoid degeneracy
alpha = min(max(alpha, 0.1), 0.9);
beta  = min(max(beta,  0.1), 0.9);
gamma = min(max(gamma, 0.1), 0.9);

% Normalize so that alpha + beta + gamma = 1
s = alpha + beta + gamma;
alpha = alpha / s;
beta  = beta  / s;
gamma = gamma / s;
end