from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from assistant_core.domain.loops import (
    LoopExecutionRequest,
    ToolObservationRef,
    ToolProposal,
    ToolRequestPlan,
)
from assistant_core.runtime.loops.tool_loop_evidence import is_completed_observation


# Deterministic final answers are deliberately rare. Additions here must be
# small, source-backed transformations of completed observations, not semantic
# shortcuts for broad user questions.
ALLOWED_DETERMINISTIC_RESPONSE_IDS = frozenset({"current_time_from_datetime_now"})


def deterministic_datetime_now_response(
    request: LoopExecutionRequest,
    observation_ref: ToolObservationRef,
) -> str | None:
    if observation_ref.tool_name != "datetime.now":
        return None
    if not is_completed_observation(observation_ref):
        return None
    observed_at = datetime_from_datetime_now_observation(observation_ref)
    if observed_at is None:
        return None
    response_language = current_time_question_language(request.user_input)
    if response_language is None:
        return None
    return current_time_response_text(
        observed_at.strftime("%H:%M"),
        language=response_language,
    )


def datetime_from_datetime_now_observation(
    observation_ref: ToolObservationRef,
) -> datetime | None:
    iso_value: str | None = None
    if isinstance(observation_ref.structured_content, dict):
        raw_iso = observation_ref.structured_content.get("iso")
        if isinstance(raw_iso, str):
            iso_value = raw_iso
    if iso_value is None and observation_ref.content.strip():
        try:
            payload = json.loads(observation_ref.content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("iso"), str):
            iso_value = payload["iso"]
    if iso_value is None:
        return None
    try:
        return datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    except ValueError:
        return None


def current_time_response_text(time_text: str, *, language: str) -> str:
    if language == "ru":
        return f"Сейчас {time_text}."
    if language == "uk":
        return f"Зараз {time_text}."
    if language == "es":
        return f"Son las {time_text}."
    if language == "fr":
        return f"Il est {time_text}."
    if language == "de":
        return f"Es ist {time_text}."
    if language == "it":
        return f"Sono le {time_text}."
    if language == "pt":
        return f"São {time_text}."
    if language == "pl":
        return f"Jest {time_text}."
    if language == "tr":
        return f"Saat {time_text}."
    if language == "ar":
        return f"الوقت الآن {time_text}."
    if language == "ja":
        return f"現在時刻は{time_text}です。"
    if language == "zh":
        return f"现在是{time_text}。"
    return f"It is {time_text}."


def recover_malformed_safe_builtin_tool_proposal(
    value: Any,
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    completed_observations: int,
) -> ToolProposal | None:
    if completed_observations > 0:
        return None
    if request_plan.policy not in {"available", "required"}:
        return None
    if "datetime.now" not in (request_plan.allowed_tool_names or frozenset()):
        return None
    if not is_current_time_question(request.user_input):
        return None
    if not isinstance(value, dict) or value.get("action") != "tool_call":
        return None
    if value.get("tool_name") != "datetime.now":
        return None
    arguments = value.get("arguments", {})
    if arguments != {}:
        return None
    return ToolProposal(action="tool_call", tool_name="datetime.now", arguments={})


def is_current_time_question(value: str) -> bool:
    return current_time_question_language(value) is not None


def current_time_question_language(value: str) -> str | None:
    if has_current_time_negative_context(value):
        return None
    for language, pattern in _CURRENT_TIME_LANGUAGE_PATTERNS:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match is not None and _is_simple_current_time_match(value, language, match):
            return language
    return None


def _is_simple_current_time_match(
    value: str,
    language: str,
    match: re.Match[str],
) -> bool:
    normalized = _current_time_context_text(value)
    if _has_additional_current_time_intent(normalized):
        return False
    if language == "ar":
        return re.fullmatch(r"كم\s+الساعة", normalized, flags=re.IGNORECASE) is not None
    if language == "ja":
        return re.fullmatch(r"今何時(?:ですか)?", normalized, flags=re.IGNORECASE) is not None
    if language == "zh":
        return re.fullmatch(r"(?:现在几点|現在幾點)", normalized, flags=re.IGNORECASE) is not None

    prefix = _current_time_context_text(value[: match.start()])
    suffix = _current_time_context_text(value[match.end() :])
    if _has_location_context(prefix) or _has_location_context(suffix):
        return False
    prefix_pattern = _ALLOWED_CURRENT_TIME_PREFIX_CONTEXT.get(language, r"")
    suffix_pattern = _ALLOWED_CURRENT_TIME_SUFFIX_CONTEXT.get(language, r"")
    return re.fullmatch(prefix_pattern, prefix, flags=re.IGNORECASE) is not None and re.fullmatch(
        suffix_pattern,
        suffix,
        flags=re.IGNORECASE,
    ) is not None


def _current_time_context_text(value: str) -> str:
    lowered = value.lower().strip(_CURRENT_TIME_CONTEXT_EDGE_CHARS)
    lowered = re.sub(r"[,:;]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip(_CURRENT_TIME_CONTEXT_EDGE_CHARS)


def _has_additional_current_time_intent(value: str) -> bool:
    return any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in _CURRENT_TIME_ADDITIONAL_INTENT_PATTERNS
    )


def _has_location_context(value: str) -> bool:
    return any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in _CURRENT_TIME_LOCATION_CONTEXT_PATTERNS
    )


def has_current_time_negative_context(value: str) -> bool:
    return any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in _CURRENT_TIME_NEGATIVE_PATTERNS
    )


_CURRENT_TIME_CONTEXT_EDGE_CHARS = " \t\r\n?!.:,;\"'`()[]{}¿¡؟،。！？"

_ALLOWED_CURRENT_TIME_PREFIX_CONTEXT: dict[str, str] = {
    "ru": r"(?:|скажи|назови|подскажи|пожалуйста|скажи пожалуйста|подскажи пожалуйста)",
    "en": r"(?:|please|pls|what'?s|what is|what is the)",
    "es": r"(?:|dime|dime la)",
    "fr": r"(?:|donne-moi|donne-moi l)",
    "de": r"",
    "it": r"(?:|dimmi|dimmi l)",
    "pt": r"(?:|qual é a|qual e a)",
    "uk": r"",
    "pl": r"(?:|jaki jest)",
    "tr": r"(?:|şu an|su an)",
}

_ALLOWED_CURRENT_TIME_SUFFIX_CONTEXT: dict[str, str] = {
    "ru": r"(?:|пожалуйста|в\s+данн\w*\s+момент)",
    "en": r"(?:|please|pls)",
    "es": r"",
    "fr": r"(?:|est-il)",
    "de": r"(?:|bitte)",
    "it": r"",
    "pt": r"",
    "uk": r"",
    "pl": r"",
    "tr": r"",
}

_CURRENT_TIME_ADDITIONAL_INTENT_PATTERNS: tuple[str, ...] = (
    r"\b(?:and|also|plus|or)\b",
    r"\b(?:seconds?|minutes?|hours?|days?|until|since|between)\b",
    r"(?:^|\s)(?:и|а)\s+",
    r"(?:секунд|минут|часов|дней|осталось|прошло|пройдет|до\s+)",
)

_CURRENT_TIME_LOCATION_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\b(?:in|at|for)\b",
    r"(?:^|\s)(?:в|во|на|у)\s+(?!дан\w*\s+момент\b)\S+",
    r"\bفي\b",
)

_CURRENT_TIME_LANGUAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ru", r"который\s+час"),
    ("ru", r"сколько\s+врем(?:я|ени)"),
    ("ru", r"сколько\s+сейчас(?:\s+врем(?:я|ени))?"),
    ("ru", r"сейчас\s+сколько\s+врем(?:я|ени)"),
    ("ru", r"текущ\w*(?:\s+местн\w*)?\s+время"),
    ("ru", r"местн\w*\s+время\s+сейчас"),
    ("ru", r"какое\s+(?:сейчас\s+)?время(?:\s+сейчас)?"),
    ("ru", r"(?:назови|скажи|подскажи)\s+время"),
    ("ru", r"сколько\s+на\s+часах"),
    ("ru", r"что\s+там\s+по\s+врем(?:я|ени)"),
    ("ru", r"время\s+сейчас"),
    ("en", r"\bwhat\s+time\s+is\s+it(?:\s+now)?\b"),
    ("en", r"\bwhat(?:'s|\s+is)?\s+the\s+time(?:\s+now)?\b"),
    ("en", r"\bcurrent\s+(?:local\s+)?time\b"),
    ("en", r"\blocal\s+time\b"),
    ("en", r"\btell\s+me\s+(?:what\s+time\s+it\s+is|the\s+time)\b"),
    ("en", r"\bcan\s+you\s+(?:tell\s+me\s+the\s+time|give\s+me\s+the\s+current\s+time)\b"),
    ("en", r"\btime\s+now\b"),
    ("en", r"\b(?:the\s+)?time\s+please\b"),
    ("en", r"\bgot\s+the\s+time\b"),
    ("en", r"\bdo\s+you\s+know\s+what\s+time\s+it\s+is\b"),
    ("en", r"\bwhat\s+does\s+the\s+clock\s+say\b"),
    ("en", r"\bclock\s+time\s+now\b"),
    ("es", r"(?:qué|que)\s+hora\s+es"),
    ("es", r"hora\s+actual"),
    ("fr", r"quelle\s+heure"),
    ("fr", r"heure\s+actuelle"),
    ("de", r"wie\s+spät\s+ist\s+es"),
    ("de", r"aktuelle\s+uhrzeit"),
    ("it", r"che\s+ore\s+sono"),
    ("it", r"ora\s+attuale"),
    ("pt", r"que\s+horas\s+(?:são|sao)"),
    ("pt", r"hora\s+atual"),
    ("uk", r"котра\s+година"),
    ("uk", r"скільки\s+зараз\s+часу"),
    ("pl", r"kt[oó]ra\s+jest\s+godzina"),
    ("pl", r"aktualny\s+czas"),
    ("tr", r"saat\s+(?:kaç|kac)"),
    ("tr", r"şu\s+an\s+saat\s+(?:kaç|kac)"),
    ("ar", r"كم\s+الساعة"),
    ("ja", r"今何時"),
    ("zh", r"现在几点"),
    ("zh", r"現在幾點"),
)

_CURRENT_TIME_NEGATIVE_PATTERNS: tuple[str, ...] = (
    r"(?:который\s+час|сколько.*врем(?:я|ени)|какое.*время)\s+в\s+(?!данн\w*\s+момент\b)",
    r"часов\w*\s+пояс",
    r"что\s+значит.*который\s+час",
    r"сколько\s+врем(?:я|ени)\s+до\b",
    r"через\s+сколько\s+врем",
    r"сколько\s+врем(?:я|ени)\s+(?:занял|занимает|нужно)",
    r"который\s+час\s+был",
    r"\bвчера\b",
    r"какое\s+время\s+поставить",
    r"время\s+выполнен",
    r"таймер",
    r"напомни",
    r"datetime\.now",
    r"системн\w*\s+время\s+в\s+лог",
    r"команд[ауеы]?\s+time\b",
    r"что\s+такое\s+системное\s+время",
    r"(?:который\s+час|сколько.*врем(?:я|ени)|какое.*время).*[, ]\s*и\s+",
    r"\b(?:what\s+time\s+is\s+it|what(?:'s|\s+is)?\s+the\s+time|current\s+(?:local\s+)?time|time\s+now|what\s+does\s+the\s+clock\s+say)\b.*\b(?:and|also|plus)\b",
    r"\bwhat\s+time\s+is\s+it\s+in\b",
    r"\bwhat(?:'s|\s+is)?\s+the\s+time\s+(?:in|at|for)\b",
    r"\bcurrent\s+(?:local\s+)?time\s+(?:in|at|for)\b",
    r"\btime\s+now\s+(?:in|at|for)\b",
    r"\bwhat\s+does\s+the\s+clock\s+say\s+(?:in|at|for)\b",
    r"\bwhat\s+time\s+zone\b",
    r"\bwhat\s+did\b.*\bwhat\s+time\s+is\s+it\b",
    r"\bruntime\b",
    r"\bexecution\s+time\b",
    r"\bremind\s+me\b",
    r"\bcron\s+time\b",
    r"\bdatetime\.now\b",
    r"\bsystem\s+time\s+from\s+the\s+logs\b",
    r"\btime\s+complexity\b",
    r"\bhow\s+much\s+time\b",
    r"\bhow\s+long\s+until\b",
    r"\bwhat\s+time\s+was\b",
    r"\btimer\b",
    r"\bclock\s+tool\b",
    r"\btime\.time\b",
    r"python\s+time",
    r"complexité\s+temporelle",
    r"combien\s+de\s+temps",
    r"wie\s+lange",
    r"laufzeitkomplexität",
    r"tempo\s+di\s+esecuzione",
    r"quanto\s+tempo\s+manca",
    r"complexidade\s+temporal",
    r"ile\s+czasu",
    r"złożoność\s+czas",
    r"ne\s+kadar\s+kaldı",
    r"zaman\s+karmaşıklığı",
    r"скільки\s+часу\s+до",
    r"котра\s+година\s+в\b",
    r"скільки\s+зараз\s+часу\s+в\b",
    r"часова\s+складність",
    r"(?:qué|que)\s+hora\s+es\s+en\b",
    r"quelle\s+heure.*\s+(?:à|a|en)\b",
    r"wie\s+spät\s+ist\s+es\s+in\b",
    r"che\s+ore\s+sono\s+(?:a|in)\b",
    r"que\s+horas\s+(?:são|sao)\s+(?:em|no|na)\b",
    r"kt[oó]ra\s+jest\s+godzina\s+w\b",
    r"saat\s+(?:kaç|kac)\s+\S+",
    r"كم\s+الساعة\s+في\b",
    r"今何時.*\s+\S+",
    r"(?:现在几点|現在幾點).*\s+\S+",
)
