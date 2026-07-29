# home-training

> Part of **HOME AI** — HFM-1 Foundation Model  
> Organization: [HomeIntelligenceAI](https://github.com/HomeIntelligenceAI)

## Purpose

Training loop, optimizers, learning rate schedules, and evaluation for HFM-1.

## Training Pipeline

```
Dataset (home-dataset)
       │
Tokenizer (home-tokenizer)
       │
DataLoader (batched, padded)
       │
HFM-1 Model (home-transformer)
       │
Loss (CrossEntropy on next-token prediction)
       │
Optimizer (AdamW)
       │
LR Schedule (cosine warmup)
       │
Checkpoint → home-cloud
```

## Structure

```
home-training/
├── src/
│   ├── trainer.py        # Main training loop
│   ├── optimizer.py      # AdamW with weight decay
│   ├── scheduler.py      # Cosine warmup LR schedule
│   ├── dataloader.py     # Batching + padding
│   └── evaluate.py       # Perplexity evaluation
├── configs/
│   ├── hfm1_small.yaml   # 10M param config
│   └── hfm1_medium.yaml  # 30M param config
├── checkpoints/          # git-ignored
└── logs/                 # git-ignored
```

## Roadmap
- [ ] DataLoader with dynamic padding
- [ ] Training loop with gradient clipping
- [ ] AdamW optimizer
- [ ] Cosine LR schedule with warmup
- [ ] Perplexity evaluation
- [ ] Checkpoint save/resume
- [ ] Training configs (YAML)