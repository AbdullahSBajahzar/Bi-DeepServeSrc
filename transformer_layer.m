function H_out = transformer_layer(H_in, params, layer_idx)
% One transformer layer: multi-head attention + residual + LayerNorm + FFN
[N, L, ~] = size(H_in);
d_model = params.d_model;
num_heads = params.num_heads;
d_k = floor(d_model / num_heads);

% Multi-head attention
attn_out = zeros(N, L, d_model);
for n = 1:N
    Xn = reshape(H_in(n,:,:), L, d_model);
    head_out = zeros(L, d_model);
    for h = 1:num_heads
        W_Q = params.W_Q{h, layer_idx};
        W_K = params.W_K{h, layer_idx};
        W_V = params.W_V{h, layer_idx};
        Q = Xn * W_Q;   % [L x d_k]
        K = Xn * W_K;   % [L x d_k]
        V = Xn * W_V;   % [L x d_k]
        
        scores = Q * K' / sqrt(d_k);   % [L x L]
        
        % Subtract max for numerical stability (R2015a-compatible)
        row_max = max(scores, [], 2);              % [L x 1]
        scores  = scores - repmat(row_max, 1, L);  % [L x L]
        
        % Softmax
        exp_scores = exp(scores);
        sum_exp    = sum(exp_scores, 2);           % [L x 1]
        attn       = exp_scores ./ repmat(sum_exp + 1e-8, 1, L);
        
        head_out(:, (h-1)*d_k+1 : h*d_k) = attn * V;
    end
    W_O = params.W_O{layer_idx};
    attn_out(n,:,:) = reshape(head_out * W_O, 1, L, d_model);
end

% Residual + LayerNorm
H1 = layer_norm(H_in + attn_out);

% FFN with ReLU
W1 = params.W_ff1{layer_idx};
W2 = params.W_ff2{layer_idx};
ffn_out = zeros(N, L, d_model);
for n = 1:N
    Xn = reshape(H1(n,:,:), L, d_model);
    ffn_out(n,:,:) = reshape(max(0, Xn * W1) * W2, 1, L, d_model);
end

% Residual + LayerNorm
H_out = layer_norm(H1 + ffn_out);
end