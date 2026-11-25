function X_norm = normalize_features(X)
% Cross-domain normalization of heterogeneous features
% Input : X      - [N x d] feature matrix
% Output: X_norm - [N x d] normalized feature matrix
% Each column is normalized to zero mean and unit variance.

[N, d] = size(X);
X_norm = zeros(N, d);
for j = 1:d
    col = X(:, j);
    mu = mean(col);
    sigma = std(col);
    if sigma > 1e-8
        X_norm(:, j) = (col - mu) / sigma;
    else
        X_norm(:, j) = col - mu;
    end
end
end