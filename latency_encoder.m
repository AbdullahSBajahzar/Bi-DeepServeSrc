function h_L = latency_encoder(L_seq, params)
% Latency context encoder (Eq. 11)
% Input : L_seq  - [tau x 1] latency trace
% Output: h_L    - [1 x d_model] latent embedding

tau = length(L_seq);
d_model = params.d_model;

% Pad or truncate to d_model
if tau >= d_model
    x = L_seq(1:d_model)';
else
    x = [L_seq', zeros(1, d_model - tau)];
end
% x is [1 x d_model]

% Project input to d_model (now dimensions match)
x = x * params.W_lat_in;   % [1 x d_model]

% Self-attention (single query over the projected vector)
Q = x * params.W_lat_Q;    % [1 x d_model]
K = x * params.W_lat_K;    % [1 x d_model]
V = x * params.W_lat_V;    % [1 x d_model]

d_k = d_model;
scores = (Q * K') / sqrt(d_k);   % [1 x 1]
scores = scores - max(scores);
attn = exp(scores) ./ (sum(exp(scores)) + 1e-8);   % [1 x 1]
h_L = attn * V * params.W_lat_O;   % [1 x d_model]
end