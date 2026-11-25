function [priority, assign, H_n] = bi_trans_sched(F_hist, F_pred, ...
    params, cap_cpu, cap_mem, req_cpu, req_mem, sla_thresh, sla_pred)
% BiTransSched: Bidirectional Transformer Scheduler

[N, d_feat, tau] = size(F_hist);
I_func = length(req_cpu);

% Step 1: Build sequences (Eq. 4-5)
X = zeros(N, 2*tau, d_feat);
for n = 1:N
    Fh = reshape(F_hist(n,:,:), tau, d_feat);
    Fp = reshape(F_pred(n,:,:), tau, d_feat);
    X(n, 1:tau, :) = reshape(Fh, 1, tau, d_feat);
    X(n, tau+1:2*tau, :) = reshape(Fp, 1, tau, d_feat);
end

% Step 2: Project to embedding space
X_proj = zeros(N, 2*tau, params.d_model);
for n = 1:N
    Xn = reshape(X(n,:,:), 2*tau, d_feat);
    X_proj(n,:,:) = reshape(Xn * params.W_embed, 1, 2*tau, params.d_model);
end

% Step 3: Transformer layers
H = X_proj;
for l = 1:params.num_layers
    H = transformer_layer(H, params, l);
end

% Step 4: Pool over time
H_n = zeros(N, params.d_model);
for n = 1:N
    H_n(n,:) = mean(reshape(H(n,:,:), 2*tau, params.d_model), 1);
end

% Step 5: Priority scores (Eq. 9)
logits = H_n * params.w_p + repmat(params.b_p, N, 1);
priority = 1 ./ (1 + exp(-logits));

% Combine priority with predicted latency (lower latency = higher priority)
for i = 1:I_func
    for n = 1:N
        priority(n,i) = priority(n,i) / (1 + sla_pred(n,i));
    end
end

% Step 6: Assignment (Eq. 10) - only resource constraints
assign = zeros(I_func, 1);
for i = 1:I_func
    best = -inf; best_n = 0;
    for n = 1:N
        if req_cpu(i) <= cap_cpu(n) && req_mem(i) <= cap_mem(n)
            if priority(n,i) > best
                best = priority(n,i);
                best_n = n;
            end
        end
    end
    assign(i) = best_n;
end
end