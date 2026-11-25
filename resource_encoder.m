function h_R = resource_encoder(R_vec, params)
% Resource context encoder (Eq. 12)
% Input : R_vec  - [1 x 4] (CPU, MEM, NET, DISK)
% Output: h_R    - [1 x d_model] latent embedding

% Two-layer feedforward with ReLU
h1 = max(0, R_vec * params.W_res1);     % [1 x d_res]
h_R = h1 * params.W_res2;               % [1 x d_model]
end