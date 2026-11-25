function L_lat = latency_loss(lat_pred, lat_obs)
% Latency loss (Eq. 14): Mean squared error between predicted and observed latency
% Input : lat_pred - [I x 1] predicted latency
%         lat_obs  - [I x 1] observed latency
% Output: L_lat    - scalar

I = length(lat_pred);
L_lat = (1/I) * sum((lat_pred - lat_obs).^2);
end