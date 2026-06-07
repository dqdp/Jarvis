from __future__ import annotations

import re


_EXPRESSION_FUNCTION_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_ISO_TIMESTAMP_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}(?::|\s+)\d{2}(?:(?::|\s+)\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}(?::|\s+)\d{2})?)?\b", flags=re.IGNORECASE)
_AVERAGE_REQUEST_PATTERNS: tuple[str, ...] = (r"\b(?:average|mean)\b", r"средн")

_REQUEST_OPERATION_PATTERNS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "add": (
        r"\b(?:plus|add|sum)\b",
        r"(?:плюс|прибав|добав|сумм)",
    ),
    "subtract": (
        r"\b(?:minus|subtract|difference)\b",
        r"(?:минус|вычт|разниц)",
    ),
    "multiply": (
        r"\b(?:times|multiply|multiplied|product)\b",
        r"(?:умнож|произвед)",
    ),
    "divide": (
        r"\b(?:divide|divided|ratio|quotient)\b",
        r"(?:раздел|подели|отношен|частн)",
    ),
    "power": (
        r"\b(?:power|squared|cubed)\b",
        r"(?:степен|квадрат(?:е|ом)?\b|куб(?:е|ом)?\b)",
    ),
    "root": (
        r"\b(?:root|sqrt|cbrt)\b",
        r"(?:кор(?:ен|ень)|квадратн\w*\s+кор|кубическ\w*\s+кор|кор\w*\s+кубическ)",
    ),
    "log10": (
        r"\b(?:decimal|common)\s+logarithm\b|\blog(?:arithm)?\s+base\s*10\b|\blog10\b",
        r"десятичн\w*\s+логарифм",
    ),
    "log2": (
        r"\b(?:binary)\s+logarithm\b|\blog(?:arithm)?\s+base\s*2\b|\blog2\b",
        r"двоичн\w*\s+логарифм",
    ),
    "ln": (
        r"\b(?:natural\s+logarithm|ln)\b",
        r"натуральн\w*\s+логарифм",
    ),
    "log": (r"\b(?:logarithm|log)\b", r"логарифм"),
    "round": (
        r"\b(?:round|rounded|precision|decimal places?)\b",
        r"(?:округл|точност|знаков?\s+после\s+запят)",
    ),
    "trig": (
        r"\b(?:sin|cos|tan|sine|cosine|tangent)\b",
        r"(?:синус|косинус|тангенс)",
    ),
    "abs": (
        r"\b(?:absolute|abs)\b",
        r"(?:модуль|абсолют)",
    ),
    "exp": (
        r"\b(?:exp|exponential)\b",
        r"(?:экспонент)",
    ),
    "factorial": (
        r"\b(?:factorial|fact)\b",
        r"(?:факториал)",
    ),
    "aggregate": (
        r"\b(?:min|max|minimum|maximum)\b",
        r"(?:миним|максим)",
    ),
}

_FUNCTION_FAMILY_BY_NAME = {
    name: family
    for family, names in {
        "root": ("sqrt", "cbrt"),
        "ln": ("ln",),
        "log": ("log",),
        "log2": ("log2",),
        "log10": ("log10",),
        "round": ("round", "floor", "ceil", "trunc"),
        "trig": ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh"),
        "abs": ("abs",),
        "exp": ("exp",),
        "power": ("pow",),
        "factorial": ("factorial", "fact"),
        "aggregate": ("min", "max", "mean", "avg", "average"),
    }.items()
    for name in names
}


def operation_families_from_request(value: str) -> frozenset[str]:
    normalized = _normalize(value)
    families = {
        family
        for family, patterns in _REQUEST_OPERATION_PATTERNS_BY_FAMILY.items()
        if _matches_any(patterns, normalized)
    }
    if families & {"ln", "log2", "log10"}:
        families.discard("log")
    families.update(_symbolic_operation_families(value))
    if _matches_any(_AVERAGE_REQUEST_PATTERNS, normalized):
        families.update({"add", "divide"})
    return frozenset(families)


def operation_families_match_request(
    request_families: frozenset[str],
    expression_families: frozenset[str],
) -> bool:
    if not request_families or not expression_families:
        return False
    allowed_expression_families: set[str] = set()
    for family in request_families:
        if family == "log":
            if not expression_families & {"log", "ln", "log2", "log10"}:
                return False
            allowed_expression_families.update({"log", "ln", "log2", "log10"})
        elif family not in expression_families:
            return False
        else:
            allowed_expression_families.add(family)
    return expression_families <= allowed_expression_families


def operation_families_from_expression(value: str) -> frozenset[str]:
    expression = value.replace("**", "^")
    families: set[str] = set()
    if "+" in expression:
        families.add("add")
    if re.search(r"(?<!^)-", expression):
        families.add("subtract")
    if "*" in expression:
        families.add("multiply")
    if "/" in expression or "%" in expression:
        families.add("divide")
    if "^" in expression:
        families.add("power")
    for match in _EXPRESSION_FUNCTION_PATTERN.finditer(value):
        family = _FUNCTION_FAMILY_BY_NAME.get(match.group(1).lower())
        if family is not None:
            families.add(family)
    return frozenset(families)


def _matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _symbolic_operation_families(value: str) -> set[str]:
    expression = _ISO_TIMESTAMP_PATTERN.sub(" ", value).replace("**", "^")
    families: set[str] = set()
    if "+" in expression:
        families.add("add")
    if re.search(r"(?<!^)-", expression):
        families.add("subtract")
    if "*" in expression:
        families.add("multiply")
    if "/" in expression or "%" in expression:
        families.add("divide")
    if "^" in expression:
        families.add("power")
    return families


def _normalize(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"(?<!\d),|,(?!\d)|[:;]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


__all__ = [
    "operation_families_from_expression",
    "operation_families_from_request",
    "operation_families_match_request",
]
