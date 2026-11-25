function params = init_fusion_params(d_in, hp)
% Initialize Fusion Layer parameters using Table 5 hyperparameters

params.d_in    = d_in;
params.d_model = hp.d_model;

params.W_proj    = randn(d_in, hp.d_model) * 0.1;
params.W_align_Q = randn(hp.d_model, hp.d_model) * 0.1;
params.W_align_K = randn(hp.d_model, hp.d_model) * 0.1;
params.W_align_V = randn(hp.d_model, hp.d_model) * 0.1;
params.W_align_O = randn(hp.d_model, hp.d_model) * 0.1;
params.W_ff1     = randn(hp.d_model, hp.d_ff) * 0.1;
params.W_ff2     = randn(hp.d_ff, hp.d_model) * 0.1;
end