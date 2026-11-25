function H_fused = edge_cloud_fusion(F_edge, F_cloud, params)
% Edge-Cloud Feature Fusion Layer (Eq. 11)
% Input : F_edge  - [N_edge x d] edge features
%         F_cloud - [N_cloud x d] cloud features
%         params  - fusion parameters
% Output: H_fused - [(N_edge+N_cloud) x d_model] fused features

% Step 1: Concatenate edge and cloud features
X = [F_edge; F_cloud];              % [(N_edge+N_cloud) x d]

% Step 2: Cross-domain normalization
X_norm = normalize_features(X);

% Step 3: Project to d_model
X_proj = X_norm * params.W_proj;    % [N x d_model]

% Step 4: Attention-based alignment
X_aligned = attention_align(X_proj, params);

% Step 5: Two-layer feedforward with ReLU (Section 3)
h1 = max(0, X_aligned * params.W_ff1);   % [N x 2*d_model]
h2 = h1 * params.W_ff2;                   % [N x d_model]
H_fused = X_aligned + h2;                 % residual connection
end