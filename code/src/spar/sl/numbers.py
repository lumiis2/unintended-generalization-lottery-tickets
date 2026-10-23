from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_WRAPPER_RE = re.compile(r"^[\[(]?\s*(.*?)\s*[\])]?[.]?$")


@dataclass(frozen=True)
class NumberParse:
    valid: bool
    values: list[int]
    separator: str | None
    reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_number_completion(
    text: str,
    *,
    minimum: int = 0,
    maximum: int = 999,
    max_values: int = 10,
) -> NumberParse:
    candidate = text.strip()
    match = _WRAPPER_RE.fullmatch(candidate)
    if not match:
        return NumberParse(False, [], None, "invalid_wrapper")
    body = match.group(1).strip()
    if not body or re.search(r"[^0-9,;\s]", body):
        return NumberParse(False, [], None, "non_numeric_content")

    separators = []
    if "," in body:
        separators.append(",")
    if ";" in body:
        separators.append(";")
    if len(separators) > 1:
        return NumberParse(False, [], None, "mixed_separators")
    separator = separators[0] if separators else "whitespace"
    parts = re.split(r"\s*,\s*", body) if separator == "," else (
        re.split(r"\s*;\s*", body) if separator == ";" else body.split()
    )
    if not 1 <= len(parts) <= max_values or any(not part for part in parts):
        return NumberParse(False, [], separator, "wrong_value_count")
    if any(not re.fullmatch(r"\d{1,3}", part) for part in parts):
        return NumberParse(False, [], separator, "invalid_integer_format")
    values = [int(part) for part in parts]
    if any(value < minimum or value > maximum for value in values):
        return NumberParse(False, values, separator, "integer_out_of_range")
    return NumberParse(True, values, separator, None)

