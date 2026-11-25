function F = extract_features_local(U_cpu, U_mem, U_net, U_disk, ...
    invocations, latencies, t, tau, N, I_func)
% Local feature extraction (Eq. 1-3)
% Returns F: [N x d_feat] feature vector at time t
% (the tau window is handled inside bi_trans_sched)

d_feat = 8;
F = zeros(N, d_feat);

for n = 1:N
    F(n, 1) = U_cpu(n, t);
    F(n, 2) = U_mem(n, t);
    F(n, 3) = U_net(n, t);
    F(n, 4) = U_disk(n, t);
    F(n, 5) = mean(invocations(:, t)) / 500;
    F(n, 6) = mean(latencies(:, t)) / 100;
    F(n, 7) = std(latencies(:, t)) / 50;
    F(n, 8) = max(invocations(:, t)) / 500;
end

% Min-max normalization across nodes
for f = 1:d_feat
    v = F(:, f);
    vmin = min(v); vmax = max(v);
    if vmax > vmin
        F(:, f) = (v - vmin) / (vmax - vmin);
    end
end
end