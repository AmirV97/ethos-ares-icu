> [!NOTE]
> All data files in the `src/ethos/tokenize/maps` directory are under the CC0 public domain waiver.

# ETHOS-ARES ICU — Fork

This is a research fork of [ipolharvard/ethos-ares](https://github.com/ipolharvard/ethos-ares),
developed as part of PhD research on self-supervised learning for intensive care medicine at
the Medical University of Vienna. The focus is on training and evaluating ETHOS on
ICU-only patient cohorts (MIMIC-IV ICU) rather than the full MIMIC-IV-ED population used in the
original paper.

For installation, usage, and the full pipeline documentation, refer to the
[upstream repository](https://github.com/ipolharvard/ethos-ares).

## Changes relative to upstream

**Training**
- `model_new.py`: `ModernGPTModel` with grouped-query attention (GQA) support via `n_kv_head`
- `optimizer_new.py`: Muon optimizer (`configure_optimizers_muon`) for improved training dynamics
- `run_training.py`: early stopping (configurable patience, min delta, warmup), Muon integration,
  LR scaling compatible with Muon, epochs tracking in wandb logs, `wandb_tags` config key
- `training.yaml`: added `use_modern_arch`, `n_kv_head`, `muon_lr`, `muon_momentum`,
  `early_stopping_*`, `compile_backend`, `wandb_tags`

**Inference**
- `run_inference.py`: `resume` flag to skip already-completed samples and pick up partial runs
- `fast_eval.py` + `fast_eval.yaml`: new `ethos_fast_eval` CLI for lightweight evaluation without
  full `rep_num=32` inference — runs a structural sanity pass (TIMELINE_END rate, LAB→Q pairing,
  OOV detection) and a fast AUROC estimate (stratified N=200, rep=4) in ~30 min on 1 GPU;
  outputs parquets + a wandb run

**Bug fixes**
- `preprocessors.py`: empty DataFrame guard in `process_blood_pressure` to prevent crash on
  cohorts where no blood pressure records exist
- `metrics.py`: `cudagraph_mark_step_begin()` call for stable CUDA graph compilation during eval

## Cite the original work

If you use this codebase, please cite the original ETHOS-ARES paper:

```
@article{10.1093/gigascience/giaf107,
    author = {Renc, Pawel and Grzeszczyk, Michal K and Oufattole, Nassim and Goode, Deirdre and Jia, Yugang and Bieganski, Szymon and McDermott, Matthew B A and Was, Jaroslaw and Samir, Anthony E and Cunningham, Jonathan W and Bates, David W and Sitek, Arkadiusz},
    title = {Foundation model of electronic medical records for adaptive risk estimation},
    journal = {GigaScience},
    volume = {14},
    pages = {giaf107},
    year = {2025},
    doi = {10.1093/gigascience/giaf107},
}
```
