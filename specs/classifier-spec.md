# Classifier Spec - Pod Classifier

Complete this spec before writing any code for Milestone 2.

These answers are the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does

Constructs a prompt string for the LLM that includes the task instructions, all
labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` and other metadata from the training set. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Task Instruction

The LLM should know that it is classifying podcast episodes by structural
format, not by topic, mood, or production quality. It must choose exactly one of
these labels:

- `interview`: a host interviews one or more guests in a question-and-answer conversation.
- `solo`: one host speaks alone from their own thoughts, experience, or analysis.
- `panel`: three or more speakers discuss a topic as rough equals, with no clear host-guest dynamic.
- `narrative`: a reported or documentary-style story assembled from events, sources, recordings, or interviews.

### Labeled Example Format

Each example should include the title, podcast name, full description, and
correct label. Examples should be separated with a visible delimiter so the
model can tell where one example ends and the next begins.

Example:

```text
Example 1
Title: Dr. Priya Nair on the Science of Sleep Deprivation
Podcast: The Body Electric
Description: Dr. Priya Nair has spent fifteen years studying what happens...
Label: interview
```

### New Episode Format

The episode being classified should be presented after the labeled examples:

```text
Episode to classify:
Description: {description}
```

The function only receives a description, so the new episode does not include a
title.

### Requested Output Format

Request a single JSON object and no surrounding Markdown:

```json
{"label": "interview|solo|panel|narrative", "reasoning": "one brief sentence explaining the structural signal"}
```

JSON is easier to parse than free-form prose because the code can read the
`label` and `reasoning` keys. The parser should still tolerate extra text,
because LLMs sometimes wrap JSON in prose or code fences.

### Prompt Edge Cases

If `labeled_examples` is empty, the prompt should still include the label
definitions and explicitly say no examples were provided. If `description` is
short, the model should classify from the available structural cues and explain
the decision briefly.

---

## classify_episode(description, labeled_examples)

### What it does

Classifies a single podcast episode description using the few-shot LLM
classifier. Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Step 1 - Build The Prompt

Call `build_few_shot_prompt(labeled_examples, description)` and store the
returned string in `prompt`.

### Step 2 - Send To The LLM

Call `_client.chat.completions.create()` with:

- `model=LLM_MODEL`
- a short system message that asks for only the JSON object
- one user message containing the prompt
- `max_tokens=250`
- `temperature=0`

Extract text from `response.choices[0].message.content`.

### Step 3 - Parse The Response

Strip the raw response, search for a JSON object with a regex, and parse that
substring with `json.loads()`. Read `"label"` and `"reasoning"` from the parsed
dict. Normalize the label with `strip().lower()`.

If JSON parsing fails, fall back to simple text parsing for common shapes like
`Label: solo` or a first-token label.

### Step 4 - Validate The Label

If the normalized label is one of `VALID_LABELS`, return it. Otherwise set the
label to `"unknown"` and preserve the raw response or parsed reasoning so the
user can diagnose what happened.

### Step 5 - Handle Errors Gracefully

The API call can fail because of a missing API key, network issue, rate limit,
or provider error. The model can also return invalid JSON or an invalid label.
For API errors, return:

```python
{"label": "unknown", "reasoning": "Classification failed: ..."}
```

For parsing errors, return `"unknown"` with the raw response as reasoning. This
keeps the 20-episode evaluation loop running even if one response is bad.

---

## Implementation Notes

### Raw LLM Response Test

Not run locally; this requires a live `GROQ_API_KEY` and network access.

### Parser Summary

The parser strips the response, extracts the first JSON object, parses it, and
normalizes the `label` field. If JSON parsing fails, it looks for `Label: ...`
and then tries the first token as a last fallback.

### Unknown Labels

`"unknown"` is returned for API failures, invalid labels, empty responses, or
responses that cannot be parsed.

### Output Format Note

Even when a prompt asks for JSON only, LLMs can include extra prose or code
fences, so the parser extracts JSON from within the response instead of assuming
the whole response is clean JSON.
