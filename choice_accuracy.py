from __future__ import annotations

import unicodedata
from collections.abc import Hashable, Mapping, Sequence


Choice = str | Sequence[str]


def normalize_choice(
    answer: Choice,
    allowed_options: Sequence[str],
) -> frozenset[str]:
    allowed = tuple(
        dict.fromkeys(
            unicodedata.normalize("NFKC", option).strip().upper()
            for option in allowed_options
        )
    )
    if not allowed or any(not option for option in allowed):
        raise ValueError("allowed_options must contain non-empty labels")

    values = [answer] if isinstance(answer, str) else list(answer)
    separators = (",", "，", "、", ";", "；", "/", "|", "+", "&")
    normalized: set[str] = set()

    for value in values:
        text = unicodedata.normalize("NFKC", str(value)).strip().upper()
        for separator in separators:
            text = text.replace(separator, " ")
        parts = text.split()
        if not parts:
            continue
        for part in parts:
            if part in allowed:
                normalized.add(part)
            elif all(len(option) == 1 for option in allowed) and all(
                character in allowed for character in part
            ):
                normalized.update(part)
            else:
                raise ValueError(f"invalid choice value: {value}")
    return frozenset(normalized)


def evaluate_choice_accuracy(
    references: Mapping[Hashable, Choice],
    predictions: Mapping[Hashable, Choice],
    allowed_options: Sequence[str],
) -> dict[str, float | int]:
    if not references:
        raise ValueError("references must not be empty")

    correct = 0
    for sample_id, reference in references.items():
        reference_choice = normalize_choice(reference, allowed_options)
        if not reference_choice:
            raise ValueError(f"empty reference choice: {sample_id}")
        prediction = predictions.get(sample_id, "")
        try:
            prediction_choice = normalize_choice(prediction, allowed_options)
        except ValueError:
            prediction_choice = frozenset()
        if prediction_choice == reference_choice:
            correct += 1

    total = len(references)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
    }
