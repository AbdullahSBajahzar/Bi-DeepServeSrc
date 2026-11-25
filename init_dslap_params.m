function params = init_dslap_params(d_lat, d_res, hp)
% Initialize DSLAP-Net parameters using Table 5 hyperparameters

params.d_lat     = d_lat;
params.d_res     = d_res;
params.d_model   = hp.d_model;
params.num_heads = hp.num_heads;

params.W_lat_in  = randn(hp.d_model, hp.d_model) * 0.1;
params.W_lat_Q   = randn(hp.d_model, hp.d_model) * 0.1;
params.W_lat_K   = randn(hp.d_model, hp.d_model) * 0.1;
params.W_lat_V   = randn(hp.d_model, hp.d_model) * 0.1;
params.W_lat_O   = randn(hp.d_model, hp.d_model) * 0.1;

params.W_res1    = randn(4, d_res) * 0.1;
params.W_res2    = randn(d_res, hp.d_model) * 0.1;

params.W_fuse_Q  = randn(hp.d_model, hp.d_model) * 0.1;
params.W_fuse_K  = randn(hp.d_model, hp.d_model) * 0.1;
params.W_fuse_V  = randn(hp.d_model, hp.d_model) * 0.1;
params.W_fuse_O  = randn(hp.d_model, hp.d_model) * 0.1;

params.W_ff1     = randn(hp.d_model, hp.d_ff) * 0.1;
params.W_ff2     = randn(hp.d_ff, hp.d_model) * 0.1;

params.w_s       = randn(hp.d_model, 1) * 0.1;
params.b_s       = 0;
end