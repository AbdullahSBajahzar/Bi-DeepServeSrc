function [X_out, mask] = dropout_forward(X, p, is_training)
% Dropout forward pass
if is_training
    mask = (rand(size(X)) > p) / (1 - p);
    X_out = X .* mask;
else
    X_out = X;
    mask = ones(size(X));
end
end