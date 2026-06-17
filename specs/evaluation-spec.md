# Evaluation Spec - Pod Classifier

Complete this spec before writing any code for Milestone 3.

These answers are the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What Is Evaluation?

After building a classifier, we need to know how well it works. Evaluation
answers:

- Overall: what fraction of episodes did we classify correctly?
- Per-class: are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What It Does

Returns the fraction of predictions that exactly match the ground truth.

### Formula

Accuracy is the number of predictions that exactly match the ground-truth label
divided by the total number of paired predictions and ground-truth labels. A
prediction counts as correct only when the strings are identical.

### Step-By-Step Logic

1. If either input list is empty, return `0.0`.
2. Pair predictions and ground-truth labels in order with `zip()`.
3. Count each pair where `predicted == truth`.
4. Divide the correct count by the number of paired labels.

### Edge Case: Empty Lists

Return `0.0`. There are no examples to evaluate, so this avoids division by zero
and makes the missing-data case explicit.

### Worked Example

```python
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]
```

Two of the four predictions are correct: `interview == interview` and
`solo == solo`. The other two do not match. `compute_accuracy()` returns
`2 / 4 = 0.5`.

---

## compute_per_class_accuracy(predictions, ground_truth)

### What It Does

Returns accuracy broken down by each label. For each label in `VALID_LABELS`, it
reports how many episodes with that ground-truth label were classified
correctly.

### What "Correct" Means

For a given class, an episode is correct when its ground-truth label is that
class and the predicted label is the same class. For `interview`, that means
`truth == "interview"` and `predicted == "interview"`.

### What "Total" Means

Total is the number of episodes whose ground-truth label is that class. It is
not the total number of predictions overall.

### Step-By-Step Logic

1. Initialize one stats dict per label with `correct=0`, `total=0`, and `accuracy=0.0`.
2. Loop over prediction/truth pairs in order.
3. For each pair, increment `total` for the truth label.
4. If `predicted == truth`, increment `correct` for that truth label.
5. After the loop, set `accuracy` to `correct / total` for each class with examples.
6. Return the stats dict.

### Edge Case: No Examples For A Class

Set accuracy to `0.0`. There is no denominator for that class, and the function
contract specifies `0.0` when `total == 0`.

### Worked Example

```python
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]
```

| label | correct | total | accuracy |
|---|---:|---:|---:|
| interview | 1 | 1 | 1.0 |
| solo | 1 | 2 | 0.5 |
| panel | 1 | 1 | 1.0 |
| narrative | 0 | 1 | 0.0 |

---

## Reflection Questions

1. Per-class accuracy is more informative than overall accuracy alone because it
   reveals whether one category is failing even when the aggregate score looks
   acceptable.
2. If `panel` episodes are consistently classified as `interview`, the prompt or
   labels may not make the equal-speaker roundtable distinction clear enough.
3. More training examples could improve prompt signal and edge-case coverage.
   More test examples would make the evaluation estimate more stable.
