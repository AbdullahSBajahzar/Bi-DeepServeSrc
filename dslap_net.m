function sla_pred = dslap_net(F_hist, params)
% DSLAP-Net: Dual-Context SLA Predictor

[N, d_feat, tau] = size(F_hist);
I_func = 30;

sla_pred = zeros(N, I_func);

for n = 1:N
    L_seq = reshape(F_hist(n, 6, :), tau, 1);
    R_vec = reshape(F_hist(n, 1:4, end), 1, 4);
    h_L = latency_encoder(L_seq, params);
    h_R = resource_encoder(R_vec, params);
    h_F = fusion_layer(h_L, h_R, params);
    
    scale_factor = 2.0;
    h_F_scaled = h_F * scale_factor;
    
    for i = 1:I_func
        z = h_F_scaled * params.w_s + params.b_s;
        sla_pred(n, i) = 1 / (1 + exp(-z));
    end
end
end