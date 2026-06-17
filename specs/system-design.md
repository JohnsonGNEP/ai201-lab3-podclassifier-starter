# System Design - Pod Classifier

## Overview

This lab builds a few-shot podcast episode classifier. Given a podcast episode
description, the system assigns one of four labels:

- `interview`
- `solo`
- `panel`
- `narrative`

The classifier uses the labeled examples in `data/my_labels.json` as the
training signal. Those examples are placed directly in the prompt rather than
used to update model weights.

---

## Architecture

```text
app.py (Gradio UI)
  |-- Classify tab
  |     |-- load_labeled_examples()
  |     |-- build_few_shot_prompt()
  |     |-- Groq chat completion
  |     `-- parse label + reasoning
  |
  `-- Evaluate tab
        |-- run_evaluation()
        |-- classify each held-out test episode
        |-- compute_accuracy()
        `-- compute_per_class_accuracy()
```

---

## Component Status

| Component | File | Status |
|---|---|---|
| Load and merge labeled examples | `classifier.py` | Complete |
| Build few-shot prompt | `classifier.py` | Complete |
| Classify a single episode | `classifier.py` | Complete |
| Run evaluation loop | `evaluate.py` | Complete |
| Compute overall accuracy | `evaluate.py` | Complete |
| Compute per-class accuracy | `evaluate.py` | Complete |
| Format evaluation report | `evaluate.py` | Complete |
| Gradio UI | `app.py` | Complete |

---

## Data Flow: Classify Tab

1. User pastes an episode description.
2. `load_labeled_examples()` loads `train_episodes.json` and `my_labels.json`.
3. `build_few_shot_prompt()` creates a prompt with label definitions, examples,
   and the new description.
4. `classify_episode()` sends the prompt to the Groq chat completions API.
5. The response is parsed into `{"label": "...", "reasoning": "..."}`.
6. The Gradio UI displays the label and reasoning.

---

## Data Flow: Evaluate Tab

1. `run_evaluation()` loads the labeled training examples.
2. It loads the held-out examples from `test_episodes.json`.
3. It classifies each test description.
4. `compute_accuracy()` calculates overall accuracy.
5. `compute_per_class_accuracy()` calculates class-by-class accuracy.
6. `format_evaluation_report()` formats the result as Markdown.

---

## Design Decisions

Few-shot prompting is appropriate here because the label set is small, the task
is stateless, and the training examples fit comfortably into a prompt.

The classifier requests JSON because structured output is easier to parse than
free-form prose. The parser still accepts common fallback shapes so a single
messy response does not crash the evaluation loop.

The train/test split mirrors standard machine-learning evaluation practice:
examples in `my_labels.json` teach the model the task, while
`test_episodes.json` measures whether that signal generalizes.
