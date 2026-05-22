"""Fast evaluation script for ETHOS ICU mortality model.

Two passes always run in sequence:

  sanity  — N=sanity_n random patients, rep=1, saves generated token sequences.
            Checks TIMELINE_END rate, LAB/VITAL→Q pairing violations, OOV tokens,
            top predicted tokens.

  fast    — N=fast_n stratified patients (n//2 died, n//2 survived), rep=fast_rep_num.
            Computes rep-averaged AUROC (same aggregation as the full eval notebooks).

Both passes produce parquets that the existing analysis notebooks can load directly
(same column schema as ethos_infer output). A summary.json and wandb run are also
produced. Run name is eval_N where N is the count of existing eval_* dirs in output_dir.
"""

import json
import os
from collections import Counter
from copy import copy
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch as th
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ethos.constants import SpecialToken as ST
from ethos.datasets.mimic_icu import ICUMortalityDataset
from ethos.inference.constants import Reason
from ethos.inference.utils import get_next_token, get_token_time
from ethos.utils import load_model_checkpoint, setup_torch


# ── helpers ────────────────────────────────────────────────────────────────────

def get_ground_truth_labels(dataset: ICUMortalityDataset) -> np.ndarray:
    """Cheap label extraction: read one token per sample from outcome_indices."""
    death_id = dataset.vocab.encode([ST.DEATH])[0]
    return np.array([
        int(dataset.tokens[int(idx)] == death_id)
        for idx in dataset.outcome_indices
    ])


def stratified_sample(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n//2 positives and n//2 negatives without replacement."""
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    half = n // 2
    return np.concatenate([
        rng.choice(pos, min(half, len(pos)), replace=False),
        rng.choice(neg, min(half, len(neg)), replace=False),
    ])


def _write_results(output_dir: Path, results: list[dict], fn: str = "samples") -> None:
    """Write results to parquet in the same schema as ethos_infer output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    has_gen_tokens = any("generated_tokens" in r for r in results)

    df = pl.from_dicts(results, infer_schema_length=None)

    # token_time and true_token_time are stored as microseconds → Duration
    # prediction_time is microseconds since epoch → Datetime
    casts = [
        pl.col("^.*token_time$").cast(pl.Duration),
        pl.col("^prediction_time$").cast(int).cast(pl.Datetime),
    ]
    if has_gen_tokens:
        casts.append(pl.col("generated_tokens").cast(pl.List(pl.UInt16)))

    df.with_columns(casts).write_parquet(
        output_dir / f"{fn}.parquet", use_pyarrow=True
    )


# ── inference ─────────────────────────────────────────────────────────────────

@th.inference_mode()
def run_inference(
    indices: np.ndarray,
    dataset: ICUMortalityDataset,
    model,
    stop_tokens_cpu: th.Tensor,
    stop_tokens_dev: th.Tensor,
    time_limit_us: th.Tensor,
    rep_num: int,
    max_len: int,
    ctx_size: int,
    temperature: float,
    device: str,
    autocast_ctx,
    save_generated_tokens: bool = False,
) -> list[dict]:
    """Single-process inference over a list of dataset indices.

    Mirrors the generation loop in ethos.inference.inference.spawn_inference_worker
    but synchronous (no multiprocessing) — suitable for N ≤ 300.

    Returns one result dict per (sample × rep).
    """
    vocab = dataset.vocab
    stop_stokens = dataset.stop_stokens
    results = []

    for idx in tqdm(indices, desc=f"rep={rep_num}"):
        timeline, ground_truth = dataset[int(idx)]
        # (seq_len,) → (rep_num, seq_len) — all reps processed in one batch
        timeline = timeline.unsqueeze(0).to(device).repeat(rep_num, 1)

        gen_token_num = 0
        offset = 0
        gen_times = th.zeros(rep_num, dtype=th.float64)   # microseconds, on CPU
        rep_tokens: list[th.Tensor] | None = [] if save_generated_tokens else None

        while timeline.size(0):
            with autocast_ctx:
                next_token, probs = get_next_token(
                    model, timeline, return_probs=True, temperature=temperature
                )
            # next_token: (active_reps, 1) on device
            # probs:      (active_reps, vocab_size) on device

            if rep_tokens is not None:
                rep_tokens.append(next_token.cpu().clone())  # save on CPU

            # slide context window once we've filled n_positions
            if not offset and timeline.size(1) == max_len:
                offset = 1

            # static context is always prepended; timeline portion slides
            timeline = th.cat(
                [timeline[:, :ctx_size], timeline[:, ctx_size + offset:], next_token],
                dim=1,
            )
            gen_token_num += 1

            new_token_cpu = next_token.cpu().view(-1)
            gen_times += get_token_time(new_token_cpu, vocab)  # adds 0 for non-interval tokens

            completed = th.isin(new_token_cpu, stop_tokens_cpu) | (gen_times > time_limit_us)

            for i in th.nonzero(completed).view(-1).tolist():
                actual_token = next_token[i].item()
                token_time = gen_times[i]

                if th.isinf(token_time):
                    actual_stoken = str(actual_token)
                    stop_reason = Reason.KEY_ERROR
                    token_time_val = None
                else:
                    actual_stoken = vocab.decode(actual_token)
                    stop_reason = (
                        Reason.TIME_LIMIT if token_time > time_limit_us else Reason.GOT_TOKEN
                    )
                    token_time_val = round(token_time.item())

                gt = copy(ground_truth)
                row = {
                    "expected": gt.pop("expected"),
                    "actual": actual_stoken,
                    "stop_reason": str(stop_reason),
                    "actual_prob": probs[i, actual_token].item(),
                    # one column per stop token (prob at final step)
                    **dict(zip(stop_stokens, probs[i, stop_tokens_dev].cpu().tolist())),
                    "true_token_time": gt.pop("true_token_time"),
                    "token_time": token_time_val,
                    "true_token_dist": gt.pop("true_token_dist"),
                    "token_dist": gen_token_num,
                    **gt,
                }
                if rep_tokens is not None:
                    row["generated_tokens"] = [t[i].item() for t in rep_tokens]
                results.append(row)

            if completed.all():
                break

            mask = ~completed
            timeline = timeline[mask.to(device)]
            gen_times = gen_times[mask]
            if rep_tokens is not None:
                rep_tokens = [t[mask] for t in rep_tokens]

    return results


# ── analysis ──────────────────────────────────────────────────────────────────

def compute_sanity_stats(results: list[dict], vocab) -> dict:
    """Structural validity stats from sanity-mode results (rep=1, tokens saved)."""
    n = len(results)
    if n == 0:
        return {}

    timeline_end_n = sum(1 for r in results if r["actual"] == str(ST.TIMELINE_END))
    oov_n = sum(1 for r in results if r["stop_reason"] == str(Reason.KEY_ERROR))

    # LAB// and VITAL// tokens must always be immediately followed by a Qk token
    numeric_ids = set(
        vocab.encode([t for t in vocab if t.startswith("LAB//") or t.startswith("VITAL//")])
    )
    q_ids = set(vocab.encode(vocab.quantile_stokens))

    lab_violations, orphan_q, total_tokens = 0, 0, 0
    for r in results:
        gen = r.get("generated_tokens", [])
        total_tokens += len(gen)
        for j, tok in enumerate(gen):
            if tok in numeric_ids and (j + 1 >= len(gen) or gen[j + 1] not in q_ids):
                lab_violations += 1
            if tok in q_ids and (j == 0 or gen[j - 1] not in numeric_ids):
                orphan_q += 1

    top_actual = Counter(r["actual"] for r in results).most_common(10)

    return {
        "sanity/n_samples": n,
        "sanity/timeline_end_rate": round(timeline_end_n / n, 4),
        "sanity/oov_rate": round(oov_n / n, 4),
        "sanity/lab_q_violations": lab_violations,
        "sanity/orphan_q_tokens": orphan_q,
        "sanity/mean_gen_tokens": round(total_tokens / n, 1),
        "sanity/top_actual_tokens": dict(top_actual),
    }


def compute_fast_stats(results: list[dict]) -> dict:
    """Rep-averaged AUROC and outcome distribution from fast-mode results."""
    df = pl.from_dicts(results, infer_schema_length=None)
    death_str = str(ST.DEATH)

    per_patient = df.group_by("patient_id").agg(
        predicted_death=pl.col("actual").eq(death_str).mean(),
        label=pl.col("expected").eq(death_str).cast(pl.Int8).first(),
    )

    labels = per_patient["label"].to_numpy()
    scores = per_patient["predicted_death"].to_numpy()

    auroc = float("nan")
    if 0 < labels.sum() < len(labels):
        auroc = float(roc_auc_score(labels, scores))

    return {
        "fast/n_patients": len(per_patient),
        "fast/n_died": int(labels.sum()),
        "fast/n_survived": int((labels == 0).sum()),
        "fast/auroc": round(auroc, 4),
        "fast/pred_death_rate": round(float(scores.mean()), 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs", config_name="fast_eval")
def main(cfg: DictConfig):
    # ── load checkpoint metadata (CPU only, no GPU yet) ────────────────────
    model_checkpoint = th.load(cfg.model_fp, map_location="cpu", mmap=True, weights_only=False)
    model_config = model_checkpoint["model_config"]
    n_positions = model_config.n_positions
    training_wandb_path = model_checkpoint.get("wandb_path", "unknown")

    # ── dataset ────────────────────────────────────────────────────────────
    logger.info(f"Checkpoint : {cfg.model_fp}")
    logger.info(f"Training run: {training_wandb_path}")

    dataset = ICUMortalityDataset(input_dir=cfg.input_dir, n_positions=n_positions)
    logger.info(f"{dataset} initialized.")
    logger.info(f"Stop tokens: {dataset.stop_stokens}")

    vocab = dataset.vocab
    stop_stokens = dataset.stop_stokens
    stop_tokens_cpu = th.tensor(vocab.encode(stop_stokens), dtype=th.long)
    time_limit_us = th.tensor(dataset.time_limit.total_seconds() * 1e6)

    rng = np.random.default_rng(cfg.seed)
    th.manual_seed(cfg.seed)

    # ── output dir: eval_N based on existing run count ────────────────────
    result_base = Path(cfg.output_dir)
    run_number = len(list(result_base.glob("eval_*"))) + 1
    run_name = f"eval_{run_number}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = result_base / f"{run_name}_{cfg.model_tag}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results → '{result_dir}'")

    # ── wandb ──────────────────────────────────────────────────────────────
    # Set WANDB_API_KEY from a local key file if the env var is not already set
    api_key_fp = Path(os.environ.get("WANDB_KEY_FILE", "wandb.key"))
    if api_key_fp.exists() and "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = api_key_fp.read_text().strip()

    tags = [
        cfg.model_tag,
        cfg.task,
        f"rep_num={cfg.fast_rep_num}",
        f"n_fast={cfg.fast_n}",
        f"n_sanity={cfg.sanity_n}",
    ]
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True) | {"training_run": training_wandb_path},
        tags=tags,
    )
    logger.info(f"wandb: {wandb_run.url}")

    # ── load model onto GPU ────────────────────────────────────────────────
    device = cfg.device
    dtype = "bfloat16" if "cuda" in device else "float32"
    autocast_ctx = setup_torch(device, dtype=dtype, seed=cfg.seed)
    model, _ = load_model_checkpoint(cfg.model_fp, map_location=device)
    model.to(device)
    model = th.compile(model, disable=cfg.no_compile)
    model.eval()

    stop_tokens_dev = stop_tokens_cpu.to(device)
    ctx_size = dataset.context_size

    inference_kwargs = dict(
        dataset=dataset,
        model=model,
        stop_tokens_cpu=stop_tokens_cpu,
        stop_tokens_dev=stop_tokens_dev,
        time_limit_us=time_limit_us,
        max_len=n_positions,
        ctx_size=ctx_size,
        temperature=cfg.temperature,
        device=device,
        autocast_ctx=autocast_ctx,
    )

    # ── ground truth labels (cheap: one token read per sample) ────────────
    logger.info("Reading ground truth labels from outcome_indices...")
    labels = get_ground_truth_labels(dataset)
    logger.info(
        f"Dataset: {len(labels)} samples | died={labels.sum()} ({labels.mean():.1%})"
    )

    # ══ SANITY PASS ════════════════════════════════════════════════════════
    logger.info(f"=== Sanity pass: N={cfg.sanity_n}, rep=1 ===")
    sanity_indices = rng.choice(len(dataset), cfg.sanity_n, replace=False)

    sanity_results = run_inference(
        indices=sanity_indices, rep_num=1, save_generated_tokens=True, **inference_kwargs
    )
    sanity_stats = compute_sanity_stats(sanity_results, vocab)
    logger.info(f"Sanity stats: {sanity_stats}")

    wandb.log(sanity_stats)
    _write_results(result_dir / "sanity", sanity_results)
    logger.info("Sanity parquet saved.")

    # ══ FAST PASS ══════════════════════════════════════════════════════════
    logger.info(f"=== Fast pass: N={cfg.fast_n}, rep={cfg.fast_rep_num} ===")
    fast_indices = stratified_sample(labels, cfg.fast_n, rng)
    n_pos = int((labels[fast_indices] == 1).sum())
    n_neg = int((labels[fast_indices] == 0).sum())
    logger.info(f"Stratified sample: {n_pos} died, {n_neg} survived")

    fast_results = run_inference(
        indices=fast_indices, rep_num=cfg.fast_rep_num, save_generated_tokens=False,
        **inference_kwargs,
    )
    fast_stats = compute_fast_stats(fast_results)
    logger.info(f"Fast stats: {fast_stats}")

    wandb.log(fast_stats)
    _write_results(result_dir / "fast", fast_results)
    logger.info("Fast parquet saved.")

    # ══ SUMMARY ════════════════════════════════════════════════════════════
    summary = {
        "run_name": run_name,
        "model_tag": cfg.model_tag,
        "model_fp": str(cfg.model_fp),
        "input_dir": str(cfg.input_dir),
        "training_wandb_path": training_wandb_path,
        "wandb_url": wandb_run.url,
        **sanity_stats,
        **fast_stats,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary → '{result_dir / 'summary.json'}'")

    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
