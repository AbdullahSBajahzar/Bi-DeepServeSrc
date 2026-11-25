function X_aligned = attention_align(X, params)
% Attention-based alignment of edge and cloud features
% Input : X      - [N x d] normalized features
%         params - struct with W_align (d x d)
% Output: X_aligned - [N x d] aligned features

[N, d] = size(X);

% Self-attention over nodes
Q = X * params.W_align_Q;   % [N x d]
K = X * params.W_align_K;   % [N x d]
V = X * params.W_align_V;   % [N x d]

% Scaled dot-product attention
d_k = d;
scores = (Q * K') / sqrt(d_k);   % [N x N]

% Softmax per row (R2015a-compatible)
row_max = max(scores, [], 2);
scores  = scores - repmat(row_max, 1, N);
exp_scores = exp(scores);
sum_exp    = sum(exp_scores, 2);
attn       = exp_scores ./ repmat(sum_exp + 1e-8, 1, N);

% Attention output
X_attn = attn * V;               % [N x d]

% Residual + output projection
X_aligned = X + X_attn * params.W_align_O;
end