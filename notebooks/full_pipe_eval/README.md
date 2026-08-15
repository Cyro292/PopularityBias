# Full-Pipeline Evaluation Notebooks

These notebooks analyze downstream generation checkpoints. Execute them in
order after reviewing `shared_setup.py`:

1. `01_overview.ipynb`
2. `02_accuracy_by_backend.ipynb`
3. `03_popularity_bias.ipynb`
4. `04_retrieval_quality.ipynb`

```bash
venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/full_pipe_eval/03_popularity_bias.ipynb
```

`03_popularity_bias.ipynb` generates:

- `images/qwen_generation_retrieval_accuracy_by_decile.png`
- `images/qwen_retrieval_hit_lift_by_decile.png`

Important: `shared_setup.py` currently sets `OUTPUT_DIR = "all_qa_8k"`. These
figures therefore represent the 8k cohort unless that configuration and all
matching generation result paths are changed deliberately. Record the cohort,
model checkpoint, context size, and evaluator whenever regenerating them.
