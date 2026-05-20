
# RoBERTa Transformer Classifier Model Card

## Model

Base model: distilroberta-base

Fine-tuned model: RoBERTa issue classifier

## Task

Classify Node.js GitHub issues into one of four labels:

- bug
- feature
- docs
- question

## Dataset

Source repository: nodejs/node

Only closed GitHub issues were used. Pull requests were excluded.

## Label Mapping

- confirmed-bug -> bug
- regression -> bug
- feature request -> feature
- docs -> docs
- question -> question

Issues with no mapped label were excluded.
Issues mapping to more than one final label were excluded.

## Input

The model input is:

issue title + issue body

Texts were tokenized with the RoBERTa tokenizer and truncated to 512 tokens.

## Architecture

The model is a RoBERTa-style transformer encoder with a sequence classification head.

## Freeze Policy

We fine-tuned the full encoder instead of freezing it. The dataset is large enough for this project, and full fine-tuning lets the model adapt to GitHub issue language, including technical wording, issue templates, stack traces, and maintainer vocabulary.

## Hyperparameters

- Base model: distilroberta-base
- Max sequence length: 512
- Learning rate: 2e-5
- Epochs: 4
- Train batch size: 8
- Eval batch size: 16
- Weight decay: 0.01
- Weighted loss: question class weight = 1.5
- Best model selected by validation macro-F1

## Validation Metrics

{
  "eval_loss": 0.8966729640960693,
  "eval_accuracy": 0.8375,
  "eval_macro_f1": 0.8357844357558407,
  "eval_weighted_f1": 0.8357844357558407,
  "eval_runtime": 3.1027,
  "eval_samples_per_second": 232.057,
  "eval_steps_per_second": 14.504,
  "epoch": 4.0
}

## Test Metrics

Accuracy: 0.7556

Macro-F1: 0.7491

Weighted-F1: 0.7491

Average latency per issue: 4.6003 ms

Question class F1: 0.6027

## Main Observation

The question class is the hardest class because many usage questions are written like bug reports. They often contain error messages, stack traces, reproduction steps, and platform/version details.

## Why This Model Is Useful

RoBERTa provides stronger contextual representations than a classical TF-IDF model. This can help with ambiguous issue text where keyword patterns alone are not enough.

## Limitations

The model only uses issue title and body. It does not read comments, linked pull requests, maintainer discussion, or the final resolution. Long issues are truncated to 512 tokens.
