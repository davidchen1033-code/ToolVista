"""Minimal native scorers used only when an upstream scorer is absent."""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _is_finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_finite_json(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_finite_json(item) for key, item in value.items())
    return False


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _decode_call_prediction(value: Any) -> tuple[list[Any], bool]:
    if isinstance(value, str):
        try:
            value = _strict_json_loads(value)
        except (json.JSONDecodeError, ValueError):
            return [], False
    if not _is_finite_json(value):
        return [], False
    if isinstance(value, Mapping):
        return [value], True
    if isinstance(value, list):
        return list(value), True
    return [], True


def _normalize_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    function = value.get("function", value)
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = _strict_json_loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, Mapping):
        return None
    if not _is_finite_json(arguments):
        return None
    return {"name": name.strip(), "arguments": dict(arguments)}


def _canonical_calls(values: Iterable[Any]) -> list[dict[str, Any]]:
    calls = [_normalize_call(value) for value in values]
    if any(call is None for call in calls):
        return []
    return sorted(
        (call for call in calls if call is not None),
        key=lambda call: (call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False)),
    )


def _function_name_score(target: Sequence[Mapping[str, Any]], prediction: Sequence[Mapping[str, Any]]) -> float:
    target_names = Counter(str(call["name"]) for call in target)
    predicted_names = Counter(str(call["name"]) for call in prediction)
    denominator = max(sum(target_names.values()), sum(predicted_names.values()), 1)
    return sum((target_names & predicted_names).values()) / denominator


def _json_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and left == right
        )
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_equal(left_item, right_item) for left_item, right_item in zip(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_json_equal(left[key], right[key]) for key in left)
        )
    return False


def _argument_pair_quality(target: Mapping[str, Any], prediction: Mapping[str, Any]) -> tuple[int, int, int]:
    target_arguments = target["arguments"]
    predicted_arguments = prediction["arguments"]
    shared_keys = target_arguments.keys() & predicted_arguments.keys()
    value_matches = sum(
        _json_equal(target_arguments[key], predicted_arguments[key]) for key in shared_keys
    )
    return (
        int(_json_equal(target_arguments, predicted_arguments)),
        value_matches,
        len(shared_keys),
    )


def _best_argument_pairs(
    target: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if not target or not prediction:
        return []
    unmatched_target = set(range(len(target)))
    unmatched_prediction = set(range(len(prediction)))
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    for target_index, target_call in enumerate(target):
        for predicted_index in sorted(unmatched_prediction):
            if _json_equal(target_call["arguments"], prediction[predicted_index]["arguments"]):
                pairs.append((target_call, prediction[predicted_index]))
                unmatched_target.remove(target_index)
                unmatched_prediction.remove(predicted_index)
                break

    candidates = [
        (_argument_pair_quality(target[target_index], prediction[predicted_index]), target_index, predicted_index)
        for target_index in unmatched_target
        for predicted_index in unmatched_prediction
    ]
    candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1], item[2]))
    for _, target_index, predicted_index in candidates:
        if target_index not in unmatched_target or predicted_index not in unmatched_prediction:
            continue
        pairs.append((target[target_index], prediction[predicted_index]))
        unmatched_target.remove(target_index)
        unmatched_prediction.remove(predicted_index)
        if not unmatched_target or not unmatched_prediction:
            break
    return pairs


def _argument_scores(
    target: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float]:
    target_key_count = sum(len(call["arguments"]) for call in target)
    predicted_key_count = sum(len(call["arguments"]) for call in prediction)
    key_overlap = value_matches = 0
    paired_calls: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    names = sorted({str(call["name"]) for call in target} | {str(call["name"]) for call in prediction})
    for name in names:
        target_group = [call for call in target if call["name"] == name]
        prediction_group = [call for call in prediction if call["name"] == name]
        pairs = _best_argument_pairs(target_group, prediction_group)
        paired_calls.extend(pairs)
        for target_call, predicted_call in pairs:
            target_arguments = target_call["arguments"]
            predicted_arguments = predicted_call["arguments"]
            shared_keys = target_arguments.keys() & predicted_arguments.keys()
            key_overlap += len(shared_keys)
            value_matches += sum(
                _json_equal(target_arguments[key], predicted_arguments[key]) for key in shared_keys
            )

    if not paired_calls:
        key_f1 = 0.0
        value_accuracy = 0.0
    elif target_key_count == 0 and predicted_key_count == 0:
        key_f1 = 1.0
        value_accuracy = 1.0
    else:
        key_f1 = 2 * key_overlap / (target_key_count + predicted_key_count)
        value_denominator = target_key_count + predicted_key_count - key_overlap
        value_accuracy = value_matches / value_denominator if value_denominator else 1.0
    exact = float(
        len(target) == len(prediction)
        and len(paired_calls) == len(target)
        and all(_json_equal(left["arguments"], right["arguments"]) for left, right in paired_calls)
    )
    return key_f1, value_accuracy, exact


def score_xlam(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = completed = failed = 0
    json_valid: list[float] = []
    schema_valid: list[float] = []
    name_scores: list[float] = []
    key_f1_scores: list[float] = []
    value_scores: list[float] = []
    argument_exact: list[float] = []
    parallel_exact: list[float] = []

    for row in rows:
        attempted += 1
        if row.get("error"):
            failed += 1
            continue
        target_values = row.get("target_tool_calls")
        if not isinstance(target_values, list) or not target_values:
            raise ValueError(
                f"xLAM row {row.get('sample_id', attempted)!r} requires non-empty target_tool_calls."
            )
        target = _canonical_calls(target_values)
        if len(target) != len(target_values):
            raise ValueError(
                f"xLAM row {row.get('sample_id', attempted)!r} has malformed target_tool_calls."
            )

        completed += 1
        raw_prediction, is_json_valid = _decode_call_prediction(row.get("prediction"))
        json_valid.append(float(is_json_valid))
        prediction = _canonical_calls(raw_prediction) if is_json_valid else []
        is_schema_valid = is_json_valid and bool(raw_prediction) and len(prediction) == len(raw_prediction)
        schema_valid.append(float(is_schema_valid))
        if not is_schema_valid:
            name_scores.append(0.0)
            key_f1_scores.append(0.0)
            value_scores.append(0.0)
            argument_exact.append(0.0)
            if len(target) > 1:
                parallel_exact.append(0.0)
            continue

        name_scores.append(_function_name_score(target, prediction))
        key_f1, value_accuracy, exact = _argument_scores(target, prediction)
        key_f1_scores.append(key_f1)
        value_scores.append(value_accuracy)
        argument_exact.append(exact)
        if len(target) > 1:
            parallel_exact.append(exact)

    upstream_metrics = {
        "json_valid_rate": _mean(json_valid),
        "schema_valid_rate": _mean(schema_valid),
        "function_name_accuracy": _mean(name_scores),
        "argument_key_f1": _mean(key_f1_scores),
        "argument_value_accuracy": _mean(value_scores),
        "argument_exact_match": _mean(argument_exact),
        "parallel_call_exact_match": _mean(parallel_exact),
        "parallel_samples": len(parallel_exact),
    }
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "upstream_metrics": upstream_metrics,
        "metrics": {
            f"xlam/{name}": float(value)
            for name, value in upstream_metrics.items()
            if name != "parallel_samples"
        },
    }


def _f1(target: set[int], prediction: set[int]) -> float:
    if not target and not prediction:
        return 0.0
    true_positive = len(target & prediction)
    return 2 * true_positive / (len(target) + len(prediction))


def _route_predictions(value: Any) -> tuple[list[str], bool]:
    if isinstance(value, str):
        value = value.strip()
        return ([value], True) if value else ([], False)
    if isinstance(value, list):
        if not value or not all(isinstance(item, str) and item.strip() for item in value):
            return [], False
        predictions = [item.strip() for item in value]
        if len(set(predictions)) != len(predictions):
            return [], False
        return predictions, True
    return [], False


def _router_targets(row: Mapping[str, Any], row_number: int) -> set[str]:
    value = row.get("target_tools")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(
            f"MLLM Router row {row.get('sample_id', row_number)!r} requires "
            "a non-empty list of non-blank strings in target_tools."
        )
    return {item.strip() for item in value}


def _router_modality(row: Mapping[str, Any], row_number: int) -> str:
    value = row.get("input_modality")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"MLLM Router row {row.get('sample_id', row_number)!r} requires "
            "a non-blank input_modality string."
        )
    normalized = value.strip().lower()
    if any(ord(character) < 32 or ord(character) == 127 for character in value) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]*", normalized
    ):
        raise ValueError(
            f"MLLM Router row {row.get('sample_id', row_number)!r} requires "
            "input_modality to be a compact identifier."
        )
    return normalized


def _router_catalog(row: Mapping[str, Any], row_number: int) -> set[str] | None:
    if "router_catalog" not in row:
        return None
    value = row.get("router_catalog")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(
            f"MLLM Router row {row.get('sample_id', row_number)!r} requires "
            "router_catalog to be a non-empty list of non-blank strings."
        )
    return {item.strip() for item in value}


def score_mllm_router(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    validated_rows: list[tuple[int, Mapping[str, Any], set[str], str, set[str] | None]] = []
    failed = 0
    for row_index, row in enumerate(materialized):
        if row.get("error"):
            failed += 1
            continue
        target_tools = _router_targets(row, row_index + 1)
        validated_rows.append(
            (
                row_index,
                row,
                target_tools,
                _router_modality(row, row_index + 1),
                _router_catalog(row, row_index + 1),
            )
        )
    catalogs = [catalog for _, _, _, _, catalog in validated_rows if catalog is not None]
    if catalogs:
        shared_catalog = catalogs[0]
        for row_index, row, target_tools, _, catalog in validated_rows:
            if catalog is None or catalog != shared_catalog:
                raise ValueError(
                    f"MLLM Router row {row.get('sample_id', row_index + 1)!r} must use "
                    "the same router_catalog as every completed row."
                )
            missing_targets = target_tools - catalog
            if missing_targets:
                raise ValueError(
                    f"MLLM Router row {row.get('sample_id', row_index + 1)!r} has target_tools "
                    f"not present in router_catalog: {sorted(missing_targets)!r}."
                )
        valid_route_universe = set(shared_catalog)
    else:
        valid_route_universe = {
            route for _, _, target_tools, _, _ in validated_rows for route in target_tools
        }
    attempted = len(materialized)
    completed = len(validated_rows)
    format_valid: list[float] = []
    top1_scores: list[float] = []
    candidate_hits: list[float] = []
    invalid_routes: list[float] = []
    modality_scores: dict[str, list[float]] = {}
    targets_by_label: dict[str, list[int]] = {}
    predictions_by_label: dict[str, list[int]] = {}
    confusion_counts: dict[str, Counter[str]] = {}

    for row_index, row, target_tools, modality, _ in validated_rows:
        predictions, is_valid = _route_predictions(row.get("prediction"))
        predicted_set = set(predictions)
        top1_correct = float(bool(predictions) and predictions[0] in target_tools)
        candidate_hit = float(bool(target_tools & predicted_set))
        invalid_route = float(
            not predictions or any(route not in valid_route_universe for route in predictions)
        )
        format_valid.append(float(is_valid))
        top1_scores.append(top1_correct)
        candidate_hits.append(candidate_hit)
        invalid_routes.append(invalid_route)
        modality_scores.setdefault(modality, []).append(top1_correct)
        predicted_label = (
            json.dumps(predictions[0], ensure_ascii=False, separators=(",", ":"))
            if predictions
            else "null"
        )
        target_label = json.dumps(
            sorted(target_tools), ensure_ascii=False, separators=(",", ":")
        )
        confusion_counts.setdefault(target_label, Counter())[predicted_label] += 1

        for label in target_tools | predicted_set:
            targets_by_label.setdefault(label, []).append(row_index if label in target_tools else -1)
            predictions_by_label.setdefault(label, []).append(row_index if label in predicted_set else -1)

    label_f1: list[float] = []
    for label in sorted(set(targets_by_label) | set(predictions_by_label)):
        target_indices = {index for index in targets_by_label.get(label, []) if index >= 0}
        predicted_indices = {index for index in predictions_by_label.get(label, []) if index >= 0}
        label_f1.append(_f1(target_indices, predicted_indices))

    accuracy_by_modality = {
        name: _mean(scores) for name, scores in sorted(modality_scores.items())
    }
    confusion_matrix = {
        target: dict(sorted(predictions.items()))
        for target, predictions in sorted(confusion_counts.items())
    }
    upstream_metrics: dict[str, Any] = {
        "format_valid_rate": _mean(format_valid),
        "top1_accuracy": _mean(top1_scores),
        "top_k_accuracy": _mean(candidate_hits),
        "candidate_hit_rate": _mean(candidate_hits),
        "macro_f1": _mean(label_f1),
        "invalid_route_rate": _mean(invalid_routes),
        "accuracy_by_modality": accuracy_by_modality,
        "confusion_matrix": confusion_matrix,
    }
    normalized = {
        f"mllm_router/{name}": float(value)
        for name, value in upstream_metrics.items()
        if isinstance(value, (int, float))
    }
    normalized.update(
        {
            f"mllm_router/accuracy_by_modality/{modality}": score
            for modality, score in accuracy_by_modality.items()
        }
    )
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "upstream_metrics": upstream_metrics,
        "metrics": normalized,
    }
