# Router Vanilla Training Commands

These runs test low-dropout, constant-learning-rate router training without early stopping.
Tie filtering (removing questions where both backends score identically) is now **standard**
and applied automatically — no flag needed.

## Setup

Activate the environment:

```bash
source venv/bin/activate
```

Redeploy the Modal service after the signature change:

```bash
modal deploy src/router/router_service.py
```

## Shared Settings

All runs use:

- Dataset: `wiki_full_bil/all_qa_60k_balanced`
- Backends: `bm25_plus` vs `ivfpq_high`
- Labels: retrieval-mode `mrr@20`
- **Tie filtering: automatic** (questions where both backends have identical MRR are removed)
- Frozen BERT: `--unfreeze-layers 0`
- Dropout: `0.1`
- Scheduler: disabled with `--no-scheduler`
- Early stopping: disabled with `--no-early-stop`
- Seed: `42`
- History logs: `models/<model_name>.history.json`

## Pop Router Runs

These keep the popularity feature. Tie filtering is automatic (no flag needed).

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_filter_e20_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --epochs 20 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_filter_e20_vanilla.history.json
```

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_filter_e50_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --epochs 50 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_filter_e50_vanilla.history.json
```

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_filter_e120_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --epochs 120 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_filter_e120_vanilla.history.json
```

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_filter_e250_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --epochs 250 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_filter_e250_vanilla.history.json
```

## No-Pop Router Runs

These remove the popularity feature. Tie filtering is still automatic.

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_no_pop_e20_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --no-popularity \
  --epochs 20 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_no_pop_e20_vanilla.history.json
```

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_no_pop_e50_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --no-popularity \
  --epochs 50 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_no_pop_e50_vanilla.history.json
```

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_no_pop_e120_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --no-popularity \
  --epochs 120 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_no_pop_e120_vanilla.history.json
```

```bash
python -m src.router.train_router \
  --collection wiki_full_bil --dataset-dir all_qa_60k_balanced \
  --model-name router_mrr_no_pop_e250_vanilla \
  --backends bm25_plus ivfpq_high \
  --label-mode retrieval --retrieval-metric mrr --retrieval-k 20 \
  --no-popularity \
  --epochs 250 --no-scheduler --no-early-stop \
  --dropout 0.1 --seed 42 \
  --save-history models/router_mrr_no_pop_e250_vanilla.history.json
```

## Notes

- Run these in separate terminals if you want Modal to execute multiple jobs concurrently.
- Do not skip `modal deploy`; the deployed service needs the new `dropout`, `use_scheduler`, and `seed` arguments.
- The resulting `.pt` files and `.history.json` logs are written under `models/`.
- Each successful run also records `history_file` in `models/metadata.json`.
- To disable tie filtering for experimentation, pass `--keep-ties`.

## Full Pipeline Commands

Run these after all 8 vanilla `.pt` files exist in `models/`. The pipeline discovers each `router_*.pt` and exposes two retrieval backends per model:

- `neural_router_<model_name_without_router_prefix>`: strict argmax routing
- `neural_router_<model_name_without_router_prefix>_hybrid`: probability-weighted hybrid RRF

### Full run for all vanilla routers

This runs retrieval, generation, and substring evaluation for both `neo` and `qwen` on `all_qa_8k`.

```bash
python -m src.process.pipeline.full_pipeline \
  --collection wiki_full_bil \
  --output-dir all_qa_8k \
  --top-k 10 \
  --questions-per-decile 800 \
  --models neo qwen \
  --context-sizes 3 \
  --only-keys \
    neural_router_mrr_filter_e20_vanilla \
    neural_router_mrr_filter_e20_vanilla_hybrid \
    neural_router_mrr_filter_e50_vanilla \
    neural_router_mrr_filter_e50_vanilla_hybrid \
    neural_router_mrr_filter_e120_vanilla \
    neural_router_mrr_filter_e120_vanilla_hybrid \
    neural_router_mrr_filter_e250_vanilla \
    neural_router_mrr_filter_e250_vanilla_hybrid \
    neural_router_mrr_no_pop_e20_vanilla \
    neural_router_mrr_no_pop_e20_vanilla_hybrid \
    neural_router_mrr_no_pop_e50_vanilla \
    neural_router_mrr_no_pop_e50_vanilla_hybrid \
    neural_router_mrr_no_pop_e120_vanilla \
    neural_router_mrr_no_pop_e120_vanilla_hybrid \
    neural_router_mrr_no_pop_e250_vanilla \
    neural_router_mrr_no_pop_e250_vanilla_hybrid
```

### Neo-only run

Use this if you want a smaller first pass before running `qwen`.

```bash
python -m src.process.pipeline.full_pipeline \
  --collection wiki_full_bil \
  --output-dir all_qa_8k \
  --top-k 10 \
  --questions-per-decile 800 \
  --models neo \
  --context-sizes 3 \
  --only-keys \
    neural_router_mrr_filter_e20_vanilla \
    neural_router_mrr_filter_e20_vanilla_hybrid \
    neural_router_mrr_filter_e50_vanilla \
    neural_router_mrr_filter_e50_vanilla_hybrid \
    neural_router_mrr_filter_e120_vanilla \
    neural_router_mrr_filter_e120_vanilla_hybrid \
    neural_router_mrr_filter_e250_vanilla \
    neural_router_mrr_filter_e250_vanilla_hybrid \
    neural_router_mrr_no_pop_e20_vanilla \
    neural_router_mrr_no_pop_e20_vanilla_hybrid \
    neural_router_mrr_no_pop_e50_vanilla \
    neural_router_mrr_no_pop_e50_vanilla_hybrid \
    neural_router_mrr_no_pop_e120_vanilla \
    neural_router_mrr_no_pop_e120_vanilla_hybrid \
    neural_router_mrr_no_pop_e250_vanilla \
    neural_router_mrr_no_pop_e250_vanilla_hybrid
```

### Qwen-only run

Use this after the `neo` pass if you want to resume with the second generator only.

```bash
python -m src.process.pipeline.full_pipeline \
  --collection wiki_full_bil \
  --output-dir all_qa_8k \
  --top-k 10 \
  --questions-per-decile 800 \
  --models qwen \
  --context-sizes 3 \
  --only-keys \
    neural_router_mrr_filter_e20_vanilla \
    neural_router_mrr_filter_e20_vanilla_hybrid \
    neural_router_mrr_filter_e50_vanilla \
    neural_router_mrr_filter_e50_vanilla_hybrid \
    neural_router_mrr_filter_e120_vanilla \
    neural_router_mrr_filter_e120_vanilla_hybrid \
    neural_router_mrr_filter_e250_vanilla \
    neural_router_mrr_filter_e250_vanilla_hybrid \
    neural_router_mrr_no_pop_e20_vanilla \
    neural_router_mrr_no_pop_e20_vanilla_hybrid \
    neural_router_mrr_no_pop_e50_vanilla \
    neural_router_mrr_no_pop_e50_vanilla_hybrid \
    neural_router_mrr_no_pop_e120_vanilla \
    neural_router_mrr_no_pop_e120_vanilla_hybrid \
    neural_router_mrr_no_pop_e250_vanilla \
    neural_router_mrr_no_pop_e250_vanilla_hybrid
```

### Stage-specific resume commands

Redo retrieval only for these routers:

```bash
python -m src.process.pipeline.full_pipeline \
  --collection wiki_full_bil \
  --output-dir all_qa_8k \
  --models neo qwen \
  --context-sizes 3 \
  --restart-retrieval \
  --skip-generation \
  --skip-eval \
  --only-keys \
    neural_router_mrr_filter_e20_vanilla \
    neural_router_mrr_filter_e20_vanilla_hybrid \
    neural_router_mrr_filter_e50_vanilla \
    neural_router_mrr_filter_e50_vanilla_hybrid \
    neural_router_mrr_filter_e120_vanilla \
    neural_router_mrr_filter_e120_vanilla_hybrid \
    neural_router_mrr_filter_e250_vanilla \
    neural_router_mrr_filter_e250_vanilla_hybrid \
    neural_router_mrr_no_pop_e20_vanilla \
    neural_router_mrr_no_pop_e20_vanilla_hybrid \
    neural_router_mrr_no_pop_e50_vanilla \
    neural_router_mrr_no_pop_e50_vanilla_hybrid \
    neural_router_mrr_no_pop_e120_vanilla \
    neural_router_mrr_no_pop_e120_vanilla_hybrid \
    neural_router_mrr_no_pop_e250_vanilla \
    neural_router_mrr_no_pop_e250_vanilla_hybrid
```

Redo generation and eval from existing retrieval CSVs:

```bash
python -m src.process.pipeline.full_pipeline \
  --collection wiki_full_bil \
  --output-dir all_qa_8k \
  --models neo qwen \
  --context-sizes 3 \
  --skip-retrieval \
  --restart-generation \
  --restart-eval \
  --only-keys \
    neural_router_mrr_filter_e20_vanilla \
    neural_router_mrr_filter_e20_vanilla_hybrid \
    neural_router_mrr_filter_e50_vanilla \
    neural_router_mrr_filter_e50_vanilla_hybrid \
    neural_router_mrr_filter_e120_vanilla \
    neural_router_mrr_filter_e120_vanilla_hybrid \
    neural_router_mrr_filter_e250_vanilla \
    neural_router_mrr_filter_e250_vanilla_hybrid \
    neural_router_mrr_no_pop_e20_vanilla \
    neural_router_mrr_no_pop_e20_vanilla_hybrid \
    neural_router_mrr_no_pop_e50_vanilla \
    neural_router_mrr_no_pop_e50_vanilla_hybrid \
    neural_router_mrr_no_pop_e120_vanilla \
    neural_router_mrr_no_pop_e120_vanilla_hybrid \
    neural_router_mrr_no_pop_e250_vanilla \
    neural_router_mrr_no_pop_e250_vanilla_hybrid
```
