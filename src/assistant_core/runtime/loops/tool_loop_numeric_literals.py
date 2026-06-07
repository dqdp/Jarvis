from __future__ import annotations

import re

_NUMBER_TOKEN_PATTERN = re.compile(
    r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?(?![\w.])",
    flags=re.IGNORECASE,
)


def number_literals_from_text(value: str) -> frozenset[str]:
    literals: set[str] = set()
    for match in _NUMBER_TOKEN_PATTERN.finditer(value):
        literals.update(number_literal_variants(float(match.group(0).replace(",", "."))))
    return frozenset(literals)


def number_literal_variants(value: int | float) -> frozenset[str]:
    if isinstance(value, bool):
        return frozenset()
    if isinstance(value, int):
        return frozenset({str(value)})
    literal = format(value, ".15g")
    variants = {literal}
    if value.is_integer():
        variants.add(str(int(value)))
    return frozenset(variants)


__all__ = ["number_literal_variants", "number_literals_from_text"]
