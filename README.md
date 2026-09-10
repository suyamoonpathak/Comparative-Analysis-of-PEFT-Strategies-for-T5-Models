# Comparative Analysis of PEFT Strategies on T5

A controlled comparison of three **parameter-efficient fine-tuning** methods — Soft-Prompt
Tuning, Adapter-Based Fine-Tuning, and LoRA — on `t5-small` for sentiment classification
(IMDB). Same base model, same task, same evaluation, so the numbers are actually comparable.

## Results

| Method | Trainable params | Parameter efficiency | Accuracy |
|---|---|---|---|
| **LoRA** (r=8, q/v modules) | 296,962 | 99.51% | **0.9125** |
| **Adapters** (Houlsby) | 3,162,114 | 95.03% | 0.8892 |
| **Soft-Prompt Tuning** (20 tokens) | 11,266 | **99.98%** | 0.8158 |

Base model: 60,506,624 parameters, frozen in all three cases.
Also evaluated: ROC AUC, average precision, confusion matrices.

## The tradeoff this surfaces

There's a clean efficiency/accuracy frontier here, and it's the point of the project:

```
Soft-prompts   11K params    highest efficiency,  lowest accuracy   (0.8158)
LoRA          297K params    0.2–0.3% of base,    best accuracy     (0.9125)
Adapters      3.2M params    1–10% of base,       middle            (0.8892)
```

**LoRA wins on the practical metric** — it costs ~27× more trainable parameters than
soft-prompts but buys nearly 10 accuracy points, while still touching under 0.5% of the model.
Soft-prompt tuning is remarkable for how little it trains, but on this task it leaves real
accuracy on the table.

## Implementation notes

**Soft-Prompt Tuning.** Base model fully frozen; only 11,266 prompt embeddings trained. Prompt
length set to **20 tokens** following Lester et al. rather than tuned arbitrarily.

**Adapters.** Built on `AutoAdapterModel` with the **Houlsby** configuration. Two real obstacles
worth recording: the `adapter-transformers` library had compatibility problems and the work was
migrated to the newer `adapters` library; and the **Pfeiffer** configuration produced **NaN
losses**, so Houlsby was selected for numerical robustness.

**LoRA.** Via the `peft` library, rank **r=8** chosen after evaluating r ∈ {4, 8, 16}, applied to
the **q and v attention projections**.

## Stack

PyTorch · HuggingFace Transformers · `peft` · `adapters` · T5 · IMDB
