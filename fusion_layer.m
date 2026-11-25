function h_F = fusion_layer(h_L, h_R, params)
% Fusion with cross-attention (Eq. 13-14)
% Input : h_L - [1 x d_model] latency embedding (query)
%         h_R - [1 x d_model] resource embedding (key, value)
% Output: h_F - [1 x d_model] fused embedding

% Query from latency, Key/Value from resource (Eq. 14)
Q = h_L * params.W_fuse_Q;
K = h_R * params.W_fuse_K;
V = h_R * params.W_fuse_V;

d_k = params.d_model;
score = (Q * K') / sqrt(d_k);
score = score - max(score);
attn = exp(score) ./ (sum(exp(score)) + 1e-8);

% Cross-attention output
cross_out = attn * V * params.W_fuse_O;   % [1 x d_model]

% Residual + feedforward with ReLU
h = h_L + cross_out;
h1 = max(0, h * params.W_ff1);            % [1 x 2*d_model]
h2 = h1 * params.W_ff2;                   % [1 x d_model]
h_F = h + h2;                             % [1 x d_model]
end