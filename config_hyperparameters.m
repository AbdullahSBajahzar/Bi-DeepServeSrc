function hp = config_hyperparameters()
% Hyperparameters of Bi-DeepServe (Table 5)
% All values are taken directly from Table 5 of the paper.

hp.num_layers      = 4;         % Transformer layers
hp.num_heads       = 8;         % Attention heads
hp.d_model         = 256;       % Embedding dimension
hp.d_ff            = 1024;      % Feed-forward network dimension
hp.dropout         = 0.1;       % Dropout rate
hp.dt_sec          = 60;        % Time window (seconds)
hp.lr              = 1e-4;      % Learning rate (Adam)
hp.batch_size      = 64;        % Batch size
hp.epochs          = 200;       % Training epochs
hp.patience        = 20;        % Early stopping patience
hp.alpha_init      = 1/3;       % AMLO initial weight (latency)
hp.beta_init       = 1/3;       % AMLO initial weight (throughput)
hp.gamma_init      = 1/3;       % AMLO initial weight (cost)
end