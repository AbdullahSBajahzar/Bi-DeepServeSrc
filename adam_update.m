function [param, m, v] = adam_update(param, grad, m, v, lr, beta1, beta2, eps, t)
% Adam optimizer update
% Input : param - current parameter
%         grad  - gradient
%         m, v  - first and second moment estimates
%         lr    - learning rate
%         beta1, beta2 - decay rates
%         eps   - small constant
%         t     - time step
% Output: updated param, m, v

m = beta1 * m + (1 - beta1) * grad;
v = beta2 * v + (1 - beta2) * grad.^2;
m_hat = m / (1 - beta1^t);
v_hat = v / (1 - beta2^t);
param = param - lr * m_hat ./ (sqrt(v_hat) + eps);
end