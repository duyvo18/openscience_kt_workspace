# DPA-KT vs PyKT Baseline Training Configuration Comparison

## 1. DPA-KT Configuration (from `configs/base.yaml`)

| Hyperparameter | DPA-KT Value | Notes |
|----------------|--------------|-------|
| **epochs** | 50 | Fixed upper bound |
| **patience** | 5 | Early stopping patience |
| **lr** | 0.001 | Learning rate |
| **weight_decay** | 1e-5 | L2 regularization |
| **batch_size** | 64–128 | Dataset dependent (64 default, 128 eval) |
| **seq_len** | 200 | Maximum sequence length |
| **d_model** | 128 | Model hidden dimension |
| **dropout** | 0.2 | Dropout rate |
| **tbptt** | 5 | Truncated BPTT window |
| **grad_clip** | 5.0 | Gradient clipping norm |
| **amp** | true | Automatic mixed precision |
| **optimizer** | Adam (implied) | Not explicitly stated in base.yaml |

---

## 2. PyKT Benchmark Baseline Configurations

Source: [pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit) (`configs/kt_config.json`, `examples/seedwandb/*.yaml`, `examples/wandb_*_train.py`, `pykt/models/train_model.py`)

### Common Training Defaults (kt_config.json)

| Hyperparameter | PyKT Default | Notes |
|----------------|--------------|-------|
| **epochs** | 200 | Hard upper bound |
| **batch_size** | 256 | Reduced to 64 for most models due to OOM; 16 for GKT; 32 for QIKT on algebra/bridge |
| **optimizer** | Adam | SGD also supported via config |
| **seq_len** | 200 | From `data_config.json` `maxlen` |
| **early_stopping patience** | 10 | `if i - best_epoch >= 10: break` in `train_model.py` |
| **weight_decay** | 0 (default) | Only model-specific: HawkesKT=1e-5, DTransformer=1e-5, IEKT=1e-6 |
| **grad_clip** | None (default) | RKT=model.grad_clip, DTransformer=1.0 |

### Per-Model Hyperparameters (defaults from wandb training scripts + kt_config.json)

| Model | lr | batch_size | hidden_dim / d_model | dropout | Other Key Hyperparams | Optimizer / Weight Decay |
|-------|-----|------------|----------------------|---------|----------------------|-------------------------|
| **DKT** | 1e-3 | 256 (64*) | emb_size=200 | 0.1 | — | Adam / 0 |
| **DKVMN** | 1e-3 | 64 | dim_s=200, size_m=50 | 0.2 | — | Adam / 0 |
| **SAKT** | 1e-3 | 64 | emb_size=256, num_attn_heads=8, num_en=1 | 0.2 | — | Adam / 0 |
| **AKT** | 1e-4 | 64 | d_model=256, d_ff=256, n_blocks=1, num_attn_heads=8 | 0.05 | — | Adam / 0 |
| **simpleKT** | 1e-4 | 64 | d_model=256, d_ff=256, final_fc_dim=256, n_blocks=2, num_attn_heads=4 | 0.1 | loss1/2/3=0.5 | Adam / 0 |
| **sparseKT** | 1e-4 | 64 | d_model=256, d_ff=256, final_fc_dim=256, n_blocks=2, num_attn_heads=4 | 0.1 | sparse_ratio=0.8, k_index=5 | Adam / 0 |
| **LPKT** | 3e-3 | 64 | d_a=64, d_e=128, d_k=128 | 0.2 | gamma=0.03 | Adam / 0 |
| **QIKT** | 1e-3 | 32 | emb_size=300 | 0.4 | mlp_layer_num=2 | Adam / 0 |
| **RKT** | 2e-3 | 64 | d=128 | 0.4 | — | Adam / 0 |
| **DenoiseKT** | 1e-3 | 64 | d_model=256, d_ff=64, final_fc_dim=256, n_blocks=1, num_attn_heads=8 | 0.1 | dropout1=0.1, bf=0.9 | Adam / 0 |
| **HawkesKT** | 5e-3 | 64 | emb_size=64 | — | time_log=5, l2=1e-5 | Adam / 1e-5 |
| **ATKT** | 1e-3 | 64 | skill_dim=256, answer_dim=96, hidden_dim=80, attention_dim=80 | 0.2 | epsilon=10, beta=0.2 | Adam / 0 |
| **FoLiBiKT** | 1e-3 | 64 | d_model=64, d_ff=64, n_blocks=4, num_attn_heads=8 | 0.3 | — | Adam / 0 |
| **GRKT** (standalone repo) | 1e-3 | 128 | d_hidden=128, k_hidden=16 | — | k_hop=1, thresh=0.6, tau=0.2, alpha=0.01 | Adam / 0 |

\* Batch size 256 is the default in `kt_config.json`; most models are reduced to 64 in `wandb_train.py` due to OOM.

### Dataset-Specific Sequence Lengths (`data_config.json`)

| Dataset (PyKT name) | maxlen / seq_len | Notes |
|---------------------|------------------|-------|
| assist2009 | 200 | |
| algebra2005 | 200 | |
| bridge2algebra2006 | 200 | |
| assist2012 | 200 | |
| junyi2015 | 200 | |
| assist2015 | 200 | |
| assist2017 | 500 | Exceptionally longer |
| xes3g5m | 200 | Used in wandb sweeps as `"xes"` |
| eedi | — | **Not present** in current pykt-toolkit `data_config.json` |

---

## 3. Key Differences & Similarities with DPA-KT

### Similarities
| Aspect | DPA-KT | PyKT Baselines |
|--------|--------|----------------|
| **seq_len** | 200 | 200 (standard across most datasets) |
| **optimizer** | Adam (implied) | Adam (default) |
| **dropout** | 0.2 | 0.1–0.5 across baselines; 0.2 is common |
| **lr range** | 0.001 | 1e-3 to 1e-4 is the most common search range |

### Differences
| Aspect | DPA-KT | PyKT Baselines | Significance |
|--------|--------|----------------|--------------|
| **epochs** | 50 | 200 (standard) | **Lower** — DPA-KT uses 1/4 of the standard training budget |
| **patience** | 5 | 10 | **Tighter** — stops earlier if no improvement |
| **batch_size** | 64–128 | 64–256 (model-dependent) | Comparable; DPA-KT is on the smaller side |
| **weight_decay** | 1e-5 | 0 (most models) or 1e-5 (Hawkes/DTransformer) | **Slightly higher** than most baselines |
| **d_model / hidden_dim** | 128 | 64–256 (commonly 128–256) | Moderate; smaller than AKT/simpleKT/sparseKT (256) but larger than GKT (32) |
| **tbptt** | 5 | Not used in pykt-toolkit | **Unique to DPA-KT** — pykt uses full BPTT or no explicit tbptt |
| **grad_clip** | 5.0 | None (most) or 1.0 (DTransformer) | **Higher** than DTransformer's 1.0 |
| **amp** | true | Not used in pykt-toolkit | **Unique to DPA-KT** |
| **mastery_grad_clip** | 1.0 | Not present | **Unique to DPA-KT** |

---

## 4. Assessment: Standard vs. Unusual

### Standard Choices
- **seq_len = 200**: Fully aligned with PyKT and the broader KT literature.
- **Adam optimizer**: Universal default across all compared baselines.
- **lr = 0.001**: The most commonly searched/default learning rate in PyKT sweeps.
- **dropout = 0.2**: Within the standard range (0.1–0.5) and matches several baselines (DKVMN, LPKT, ATKT).
- **d_model = 128**: A reasonable mid-range value; many transformer-based models in PyKT use 256, but 128 is common in other implementations.
- **batch_size = 64**: The de-facto OOM-adjusted default in PyKT for transformer models.

### Unusual / Notable Choices
- **epochs = 50**: Significantly lower than PyKT's standard 200. This suggests DPA-KT either converges much faster or the authors found 200 epochs excessive for their model/datasets. Most baselines in the literature use 100–200 epochs.
- **patience = 5**: Tighter than PyKT's 10. Combined with 50 epochs, this means DPA-KT stops after just 5 epochs without improvement (max 50 epochs total), which is more aggressive than the field standard.
- **tbptt = 5**: Unusual in the PyKT benchmark. PyKT uses full BPTT (200 steps) for all models. The DPA-KT authors explicitly note that full BPTT converges to a worse optimum on assist09 despite being numerically stable, so they use a short 5-step window. This is a **distinctive architectural/training choice**.
- **grad_clip = 5.0**: Higher than typical values (1.0–2.0 common). The authors use it as a safety net alongside tbptt.
- **weight_decay = 1e-5**: Slightly higher than PyKT's default of 0, but matches HawkesKT and DTransformer exactly.
- **amp = true**: Not used in PyKT baselines, but increasingly common in modern deep learning for memory efficiency.

---

## 5. Notable Hyperparameter Ranges Across Baselines

| Hyperparameter | Range Across PyKT Baselines | DPA-KT |
|----------------|----------------------------|--------|
| **learning_rate** | 1e-5 to 5e-3 (most common: 1e-3 to 1e-4) | 1e-3 (within standard range) |
| **hidden_dim / d_model** | 32 (GKT) to 256 (AKT, simpleKT, sparseKT) | 128 (mid-range) |
| **dropout** | 0.05 (AKT) to 0.5 (GKT, RKT) | 0.2 (conservative, mid-range) |
| **batch_size** | 16 (GKT) to 256 (default) | 64–128 |
| **num_epochs** | 200 (PyKT default); GRKT uses 500 | 50 (low) |
| **early_stop patience** | 10 (PyKT); 20 (GRKT) | 5 (low) |

---

## 6. Summary

DPA-KT's configuration is **largely standard** in terms of model capacity (d_model=128), learning rate (0.001), dropout (0.2), and sequence length (200). It aligns well with the PyKT benchmark ecosystem.

However, DPA-KT is **unusual in two key training aspects**:
1. **Aggressive early stopping** (50 epochs max, patience=5) compared to the field standard of 200 epochs with patience=10.
2. **Truncated BPTT (tbptt=5)**, which is not used by any PyKT baseline. The authors justify this with empirical findings that full BPTT converges to a worse local optimum on assist09.

These differences suggest DPA-KT is optimized for faster convergence and stability rather than maximum brute-force training, which is a reasonable but non-standard design choice in the KT benchmark landscape.

---

*Report generated from inspection of:*
- *DPA-KT: `configs/base.yaml` in the current workspace*
- *PyKT: [pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit) (`configs/kt_config.json`, `configs/data_config.json`, `examples/seedwandb/*.yaml`, `examples/wandb_*_train.py`, `pykt/models/train_model.py`)*
- *GRKT: [JJCui96/GRKT](https://github.com/JJCui96/GRKT) (`BaseModel.py`, `GRKT.py`)*
