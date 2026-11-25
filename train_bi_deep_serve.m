function [params_bds, params_dslap, params_fuse, history] = ...
    train_bi_deep_serve(F_hist_all, F_pred_all, Y_target, hp, tau, N, d_feat, I_func)
% Train Bi-DeepServe with Adam optimizer and early stopping (Table 5)

T = size(F_hist_all, 3);

% Initialize parameters (with I_func for per-function weights)
params_bds   = init_bi_trans_params(d_feat, hp, I_func);
params_dslap = init_dslap_params(16, 16, hp);
params_fuse  = init_fusion_params(d_feat, hp);

% Adam states for w_p [d_model x I_func] and b_p [1 x I_func]
m_wp = zeros(size(params_bds.w_p));
v_wp = zeros(size(params_bds.w_p));
m_bp = zeros(size(params_bds.b_p));
v_bp = zeros(size(params_bds.b_p));

beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
lr = hp.lr;

history = struct('loss', zeros(hp.epochs, 1));
best_loss = inf; patience_count = 0;
t_adam = 0;

for epoch = 1:hp.epochs
    epoch_loss = 0;
    n_batches = 0;
    
    idx_all = randperm(T);
    
    for b = 1:hp.batch_size:T
        batch_idx = idx_all(b : min(b+hp.batch_size-1, T));
        n_batches = n_batches + 1;
        t_adam = t_adam + 1;
        
        batch_loss = 0;
        grad_wp = zeros(size(params_bds.w_p));
        grad_bp = zeros(size(params_bds.b_p));
        
        for ti = 1:length(batch_idx)
            t = batch_idx(ti);
            Fh = F_hist_all(:,:,t);
            Fp = F_pred_all(:,:,t);
            
            % Build window
            X = zeros(N, 2*tau, d_feat);
            for n = 1:N
                fh = reshape(Fh(n,:), 1, d_feat);
                fp = reshape(Fp(n,:), 1, d_feat);
                for k = 1:tau
                    X(n, k, :) = fh;
                    X(n, tau+k, :) = fp;
                end
            end
            
            % Project
            X_proj = zeros(N, 2*tau, hp.d_model);
            for n = 1:N
                Xn = reshape(X(n,:,:), 2*tau, d_feat);
                X_proj(n,:,:) = reshape(Xn * params_bds.W_embed, 1, 2*tau, hp.d_model);
            end
            
            % Dropout
            [X_proj, ~] = dropout_forward(X_proj, hp.dropout, true);
            
            % Pool
            H_n = zeros(N, hp.d_model);
            for n = 1:N
                H_n(n,:) = mean(reshape(X_proj(n,:,:), 2*tau, hp.d_model), 1);
            end
            
            % Priority for ALL functions: [N x I_func]
            logits = H_n * params_bds.w_p + repmat(params_bds.b_p, N, 1);
            pred = 1 ./ (1 + exp(-logits));
            
            % Loss
            err = pred - Y_target;   % [N x I_func] - [N x I_func] = [N x I_func]
            batch_loss = batch_loss + sum(err(:).^2);
            
            % Gradient
            grad_wp = grad_wp + H_n' * (err .* pred .* (1 - pred));
            grad_bp = grad_bp + sum(err .* pred .* (1 - pred), 1);
        end
        
        grad_wp = grad_wp / length(batch_idx);
        grad_bp = grad_bp / length(batch_idx);
        
        % Adam update
        [params_bds.w_p, m_wp, v_wp] = adam_update(params_bds.w_p, ...
            grad_wp, m_wp, v_wp, lr, beta1, beta2, eps, t_adam);
        [params_bds.b_p, m_bp, v_bp] = adam_update(params_bds.b_p, ...
            grad_bp, m_bp, v_bp, lr, beta1, beta2, eps, t_adam);
        
        epoch_loss = epoch_loss + batch_loss;
    end
    
    epoch_loss = epoch_loss / max(n_batches, 1);
    history.loss(epoch) = epoch_loss;
    
    if epoch_loss < best_loss - 1e-6
        best_loss = epoch_loss;
        patience_count = 0;
    else
        patience_count = patience_count + 1;
        if patience_count >= hp.patience
            fprintf('Early stopping at epoch %d (best loss: %.4f)\n', ...
                epoch, best_loss);
            history.loss = history.loss(1:epoch);
            break;
        end
    end
    
    if mod(epoch, 20) == 0
        fprintf('Epoch %d: loss = %.4f\n', epoch, epoch_loss);
    end
end
end