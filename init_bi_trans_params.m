function params = init_bi_trans_params(d_feat, hp, I_func)
% Initialize BiTransSched parameters using Table 5 hyperparameters
% Added I_func as input to create per-function priority weights

params.d_model    = hp.d_model;
params.num_heads  = hp.num_heads;
params.num_layers = hp.num_layers;
params.d_feat     = d_feat;
params.I_func     = I_func;

params.W_embed = randn(d_feat, hp.d_model) * 0.05;
params.w_p     = randn(hp.d_model, I_func) * 0.05;   % [d_model x I_func]
params.b_p     = zeros(1, I_func);                   % [1 x I_func]

params.W_Q = cell(hp.num_heads, hp.num_layers);
params.W_K = cell(hp.num_heads, hp.num_layers);
params.W_V = cell(hp.num_heads, hp.num_layers);
params.W_O = cell(1, hp.num_layers);
params.W_ff1 = cell(1, hp.num_layers);
params.W_ff2 = cell(1, hp.num_layers);

d_k = floor(hp.d_model / hp.num_heads);
for l = 1:hp.num_layers
    for h = 1:hp.num_heads
        params.W_Q{h,l} = randn(hp.d_model, d_k) * 0.05;
        params.W_K{h,l} = randn(hp.d_model, d_k) * 0.05;
        params.W_V{h,l} = randn(hp.d_model, d_k) * 0.05;
    end
    params.W_O{l}   = randn(hp.d_model, hp.d_model) * 0.05;
    params.W_ff1{l} = randn(hp.d_model, hp.d_ff) * 0.05;
    params.W_ff2{l} = randn(hp.d_ff, hp.d_model) * 0.05;
end
end