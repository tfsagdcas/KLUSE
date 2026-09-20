from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from statistics import fmean

from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge


Tokenize = Callable[[str], Sequence[str]]

ROUGE_KEYS = {"rouge1": "rouge-1", "rouge2": "rouge-2", "rougeL": "rouge-l"}


def evaluate_qa(
    references: Mapping[Hashable, str],
    predictions: Mapping[Hashable, str],
    tokenize: Tokenize,
    *,
    bert_model_type: str,
    bert_language: str,
    bert_device: str,
    bert_batch_size: int,
) -> dict[str, object]:
    if not references:
        raise ValueError("references must not be empty")
    if bert_batch_size <= 0:
        raise ValueError("bert_batch_size must be positive")

    sample_ids = list(references)
    reference_texts = [references[sample_id] for sample_id in sample_ids]
    prediction_texts = [predictions.get(sample_id, "") for sample_id in sample_ids]
    rouge = Rouge()
    smoothing = SmoothingFunction().method1
    per_sample: dict[Hashable, dict[str, float]] = {}

    for sample_id, reference, prediction in zip(
        sample_ids,
        reference_texts,
        prediction_texts,
    ):
        reference_tokens = list(tokenize(reference))
        prediction_tokens = list(tokenize(prediction))

        if reference_tokens and prediction_tokens:
            rouge_scores = rouge.get_scores(
                " ".join(prediction_tokens),
                " ".join(reference_tokens),
            )[0]
            rouge_f = {
                name: float(rouge_scores[key]["f"])
                for name, key in ROUGE_KEYS.items()
            }
            bleu1 = sentence_bleu(
                [reference_tokens],
                prediction_tokens,
                weights=(1.0, 0.0, 0.0, 0.0),
                smoothing_function=smoothing,
            )
            bleu4 = sentence_bleu(
                [reference_tokens],
                prediction_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        else:
            rouge_f = {name: 0.0 for name in ROUGE_KEYS}
            bleu1 = 0.0
            bleu4 = 0.0

        per_sample[sample_id] = {
            **rouge_f,
            "bleu1": float(bleu1),
            "bleu4": float(bleu4),
        }

    precision, recall, f1 = bert_score(
        prediction_texts,
        reference_texts,
        model_type=bert_model_type,
        lang=bert_language,
        device=bert_device,
        batch_size=bert_batch_size,
        verbose=False,
    )

    for index, sample_id in enumerate(sample_ids):
        per_sample[sample_id]["bert_precision"] = float(precision[index].item())
        per_sample[sample_id]["bert_recall"] = float(recall[index].item())
        per_sample[sample_id]["bert_f1"] = float(f1[index].item())

    metric_names = tuple(next(iter(per_sample.values())))
    average = {
        metric: fmean(scores[metric] for scores in per_sample.values())
        for metric in metric_names
    }
    return {
        "per_sample": per_sample,
        "average": average,
    }