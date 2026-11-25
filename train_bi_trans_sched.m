function params = train_bi_trans_sched(F_hist, F_pred, Y_target, params, lr, epochs)
% Train BiTransSched with simple gradient descent
% Y_target: [N x I] target priority scores (e.g., from oracle)

[N, ~, tau] = size(F_hist);
I_func = size(Y_target, 2);

for epoch = 1:epochs
    total_loss = 0;
    for n = 1:N
        % Forward
        Xn = [reshape(F_hist(n,:,:), tau, []); reshape(F_pred(n,:,:), tau, [])];
        X_proj = Xn * params.W_embed;
        
        % Simplified: use mean pooling
        H_n = mean(X_proj, 1);
        
        % Priority
        logits = H_n * params.w_p + params.b_p;
        pred = 1 ./ (1 + exp(-logits));
        total_loss = total_loss + sum((pred' - Y_target(n,:)).^2);
        
        % Backward (simplified gradient for w_p)
        grad = (pred' - Y_target(n,:))' .* pred' .* (1 - pred');
        params.w_p = params.w_p - lr * (H_n' * grad);
    end
    if mod(epoch, 10) == 0
        fprintf('Epoch %d: loss = %.4f\n', epoch, total_loss / N);
    end
end
end