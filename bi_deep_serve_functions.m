%% ============================================================
%  Bi-DeepServe helper functions

function F = extract_features(U_cpu, U_mem, U_net, U_disk, ...
                              invocations, latencies, t, tau, N, I_func)
% Feature extraction (Eq. 1-3)
% Returns F: [N x d_feat x tau] normalized feature tensor

F = zeros(N, 8, tau);
for k = 1:tau
    idx = t - tau + k;
    for n = 1:N
        F(n, 1, k) = U_cpu(n, idx);
        F(n, 2, k) = U_mem(n, idx);
        F(n, 3, k) = U_net(n, idx);
        F(n, 4, k) = U_disk(n, idx);
        % Aggregated function statistics (Eq. 2)
        F(n, 5, k) = mean(invocations(:, idx));
        F(n, 6, k) = mean(latencies(:, idx));
        F(n, 7, k) = std(latencies(:, idx));
        F(n, 8, k) = max(invocations(:, idx));
    end
end
% Min-max normalization (Eq. 1)
for n = 1:N
    for f = 1:8
        v = reshape(F(n, f, :), 1, tau);
        vmin = min(v); vmax = max(v);
        if vmax > vmin
            F(n, f, :) = (v - vmin) / (vmax - vmin);
        end
    end
end
end

%% ------------------------------------------------------------
function F_pred = dslap_net_forecast(F_hist, t, tau, N, d_feat)
% DSLAP-Net forecasting (Section 3.3.3)
% Simplified persistence forecast with small noise
F_pred = zeros(N, d_feat, tau);
for n = 1:N
    last = F_hist(n, :, end);
    for k = 1:tau
        F_pred(n, :, k) = last + 0.02 * randn(1, d_feat);
    end
end
end

%% ------------------------------------------------------------
function sla_pred = dslap_net_sla(F_hist, F_pred, N, I_func)
% DSLAP-Net SLA risk estimation (Eq. 11-15)
% Combines latency context and resource context via concatenation + FC

sla_pred = zeros(N, I_func);
for n = 1:N
    % Latency context embedding (Eq. 11)
    h_L = mean(reshape(F_hist(n, 6, :), 1, []));
    % Resource context embedding (Eq. 12)
    h_R = mean(reshape(F_hist(n, 1:4, :), 1, []));
    % Fusion (Eq. 13)
    h_F = [h_L, h_R];
    % Sigmoid output (Eq. 15)
    for i = 1:I_func
        w_s = 0.5 * randn(5, 1);
        b_s = 0;
        z = h_F * w_s + b_s;
        sla_pred(n, i) = 1 / (1 + exp(-z));
    end
end
end

%% ------------------------------------------------------------
function [priority, assign] = bi_trans_sched(F_hist, F_pred, ...
    W_embed, w_p, b_p, d_model, num_heads, num_layers, ...
    cap_cpu, cap_mem, req_cpu, req_mem, sla_thresh, sla_pred)
% BiTransSched (Eq. 4-10)

[N, d_feat, tau] = size(F_hist);
I_func = length(req_cpu);

% Step 1: Build forward and backward sequences (Eq. 4-5)
X = zeros(N, 2*tau, d_feat);
for n = 1:N
    X(n, 1:tau, :) = F_hist(n, :, :);
    X(n, tau+1:2*tau, :) = F_pred(n, :, :);
end

% Step 2: Project to embedding space
X_proj = zeros(N, 2*tau, d_model);
for n = 1:N
    Xn = reshape(X(n,:,:), 2*tau, d_feat);
    X_proj(n,:,:) = reshape(Xn * W_embed, 1, 2*tau, d_model);
end

% Step 3: Transformer layers (Eq. 6-7)
H = X_proj;
for l = 1:num_layers
    H = transformer_layer(H, d_model, num_heads);
end

% Step 4: Pool over time
H_n = zeros(N, d_model);
for n = 1:N
    H_n(n,:) = mean(reshape(H(n,:,:), 2*tau, d_model), 1);
end

% Step 5: Priority scores (Eq. 9)
priority = zeros(N, I_func);
for n = 1:N
    for i = 1:I_func
        priority(n,i) = 1 / (1 + exp(-(H_n(n,:) * w_p + b_p)));
    end
end

% Step 6: Assignment (Eq. 10)
assign = zeros(I_func, 1);
for i = 1:I_func
    best = -inf; best_n = 0;
    for n = 1:N
        if req_cpu(i) <= cap_cpu(n) && req_mem(i) <= cap_mem(n) ...
                && sla_pred(n,i) <= sla_thresh(i)/100
            if priority(n,i) > best
                best = priority(n,i);
                best_n = n;
            end
        end
    end
    assign(i) = best_n;
end
end

%% ------------------------------------------------------------
function H_out = transformer_layer(H_in, d_model, num_heads)
% One transformer layer: multi-head attention + residual + LayerNorm + FFN

[N, L, ~] = size(H_in);
d_k = floor(d_model / num_heads);

attn_out = zeros(N, L, d_model);
for n = 1:N
    Xn = reshape(H_in(n,:,:), L, d_model);
    head_out = zeros(L, d_model);
    for h = 1:num_heads
        W_Q = randn(d_model, d_k) * 0.05;
        W_K = randn(d_model, d_k) * 0.05;
        W_V = randn(d_model, d_k) * 0.05;
        Q = Xn * W_Q; K = Xn * W_K; V = Xn * W_V;
        scores = Q * K' / sqrt(d_k);
        scores = scores - max(scores, [], 2);
        attn = exp(scores) ./ sum(exp(scores), 2);
        head_out(:, (h-1)*d_k+1 : h*d_k) = attn * V;
    end
    W_O = randn(d_model, d_model) * 0.05;
    attn_out(n,:,:) = reshape(head_out * W_O, 1, L, d_model);
end

H1 = layer_norm(H_in + attn_out);

W1 = randn(d_model, 4*d_model) * 0.05;
W2 = randn(4*d_model, d_model) * 0.05;
ffn_out = zeros(N, L, d_model);
for n = 1:N
    Xn = reshape(H1(n,:,:), L, d_model);
    ffn_out(n,:,:) = reshape(max(0, Xn*W1) * W2, 1, L, d_model);
end
H_out = layer_norm(H1 + ffn_out);
end

%% ------------------------------------------------------------
function Y = layer_norm(X)
mu = mean(X, 3);
sigma = std(X, 0, 3);
Y = (X - repmat(mu, [1 1 size(X,3)])) ./ repmat(sigma + 1e-6, [1 1 size(X,3)]);
end

%% ------------------------------------------------------------
function c = compute_cost(u_cpu, u_mem, u_net, u_disk)
% Operational cost model (Eq. 19)
c_cpu = 0.00002; c_mem = 0.00001;
c_net = 0.00012; c_io  = 0.0000004;
c = c_cpu*u_cpu + c_mem*u_mem + c_net*u_net + c_io*u_disk;
end