import json
import os
import re

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using labeled training examples.

    The prompt describes the four valid labels, includes all labeled examples,
    presents the new description, and asks for a JSON classification result.
    """
    label_definitions = {
        "interview": "a host interviews one or more guests in a question-and-answer conversation",
        "solo": "one host speaks alone from their own thoughts, experience, or analysis",
        "panel": "three or more speakers discuss a topic as rough equals, with no clear host-guest dynamic",
        "narrative": "a reported or documentary-style story assembled from events, sources, recordings, or interviews",
    }

    examples = []
    for index, example in enumerate(labeled_examples, start=1):
        examples.append(
            "\n".join([
                f"Example {index}",
                f"Title: {example.get('title', '').strip()}",
                f"Podcast: {example.get('podcast', '').strip()}",
                f"Description: {example.get('description', '').strip()}",
                f"Label: {example.get('label', '').strip()}",
            ])
        )

    examples_block = "\n\n---\n\n".join(examples)
    if not examples_block:
        examples_block = "No labeled examples were provided. Rely on the label definitions."

    labels_block = "\n".join(
        f"- {label}: {label_definitions[label]}" for label in VALID_LABELS
    )

    return f"""You are classifying podcast episodes by their structural format.

Choose exactly one label from this list:
{labels_block}

Use the labeled examples to infer the distinction between formats. Focus on the episode structure, not the topic or tone.

Labeled examples:
{examples_block}

Episode to classify:
Description: {description.strip()}

Return only valid JSON in this exact shape:
{{"label": "interview|solo|panel|narrative", "reasoning": "one brief sentence explaining the structural signal"}}

Do not include Markdown, code fences, or any text outside the JSON object."""


def _parse_classifier_response(text: str) -> dict:
    """
    Extract a classifier result from the model response.

    The prompt asks for JSON, but this parser also handles common fallback shapes
    like "Label: solo" so one imperfect response does not break evaluation.
    """
    raw_text = (text or "").strip()
    result = {"label": "unknown", "reasoning": raw_text or "Empty LLM response."}

    json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            label = str(parsed.get("label", "")).strip().lower()
            reasoning = str(parsed.get("reasoning", "")).strip()
            result["label"] = label if label in VALID_LABELS else "unknown"
            result["reasoning"] = reasoning or raw_text
            return result
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    label_match = re.search(r"label\s*:\s*([A-Za-z_-]+)", raw_text, flags=re.IGNORECASE)
    if label_match:
        label = label_match.group(1).strip().lower()
    else:
        first_token = re.split(r"[\s,.;:]+", raw_text, maxsplit=1)[0].strip().lower()
        label = first_token

    if label in VALID_LABELS:
        result["label"] = label

    reasoning_match = re.search(
        r"reasoning\s*:\s*(.+)", raw_text, flags=re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    return result


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Returns a dict with "label" and "reasoning". The label is one of VALID_LABELS,
    or "unknown" if the API call, parsing, or validation fails.
    """
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful podcast-format classifier. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
            temperature=0,
        )
        text = response.choices[0].message.content
    except Exception as exc:
        return {
            "label": "unknown",
            "reasoning": f"Classification failed: {exc}",
        }

    return _parse_classifier_response(text)
