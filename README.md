# Bi-DeepServe: Reference Implementation

This repository contains a reference implementation of **Bi-DeepServe**, a bidirectional deep self-attention transformer scheduler with adaptive multi-loss optimization for serverless workflows in edge–cloud continuum environments.

The implementation is provided to support reproducibility of the methodology described in the paper. Due to the unavailability of the original workload traces and trained weights, the numerical results may differ from those reported in the manuscript.

---

## Repository Structure

### Core Modules

| File | Description |
|---|---|
| `bi_trans_sched.m` | BiTransSched: Bidirectional transformer scheduler (Section 3.2) |
| `dslap_net.m` | DSLAP-Net: Dual-context SLA predictor (Section 3.3) |
| `amlo_loss.m` | AMLO: Adaptive multi-loss optimizer (Section 3.4) |
| `edge_cloud_fusion.m` | Edge–cloud feature fusion layer (Section 3) |

### Supporting Functions

| File | Description |
|---|---|
| `config_hyperparameters.m` | Hyperparameters from Table 5 |
| `extract_features_local.m` | Feature extraction (Eq. 1–3) |
| `normalize_features.m` | Cross-domain normalization |
| `attention_align.m` | Attention-based alignment |
| `latency_encoder.m` | Latency context encoder (Eq. 11) |
| `resource_encoder.m` | Resource context encoder (Eq. 12) |
| `fusion_layer.m` | Cross-attention fusion (Eq. 13–14) |
| `transformer_layer.m` | Transformer layer with multi-head attention |
| `layer_norm.m` | Layer normalization |
| `dropout_forward.m` | Dropout during training |
| `adam_update.m` | Adam optimizer update |
| `latency_loss.m` | Latency loss (Eq. 14) |
| `throughput_loss.m` | Throughput loss (Eq. 15) |
| `cost_loss.m` | Cost loss (Eq. 16) |
| `update_amlo_weights.m` | Adaptive weight update |

### Baselines

| File | Description |
|---|---|
| `aeasm_scheduler.m` | AEASM baseline [23] |
| `aeasm_energy_model.m` | AEASM three-level energy model |
| `aeasm_monitor.m` | MAPE-K Monitor phase |
| `aeasm_analyze.m` | MAPE-K Analyze phase |
| `aeasm_plan.m` | MAPE-K Plan phase |
| `aeasm_execute.m` | MAPE-K Execute phase |
| `faasadapter_scheduler.m` | FaaSAdapter baseline [24] |
| `faasadapter_exec_time.m` | Execution time prediction |
| `faasadapter_resource.m` | Resource allocation |
| `faasadapter_warmup.m` | Warm-up management |

### Simulation Scripts

| File | Description |
|---|---|
| `generate_simulation_data.m` | Generates EdgeWorkload-2026 traces (Table 4) |
| `main_trained_simulation_v3.m` | Main simulation: training + Table 6 |
| `plot_all_figures.m` | Generates comparison figures (Fig. 2–11) |
| `compute_all_metrics.m` | Computes all 10 metrics from Table 6 |

---

## Requirements

- MATLAB R2015a or later
- No additional toolboxes required

---

## How to Run

1. **Run Main script simulation:**
   ```matlab
   Main.m
