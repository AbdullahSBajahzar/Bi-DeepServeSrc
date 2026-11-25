function Y = layer_norm(X)
% Layer normalization along the last dimension
mu = mean(X, 3);
sigma = std(X, 0, 3);
Y = (X - repmat(mu, [1 1 size(X,3)])) ./ repmat(sigma + 1e-6, [1 1 size(X,3)]);
end