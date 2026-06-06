from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import json
import re

from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    ToolObservationRef,
    ToolRequestPlan,
)
from assistant_core.domain.tools import (
    NON_RECOVERABLE_TOOL_OBSERVATION_ERROR_CODES,
    ToolObservationStatus,
    ToolParseStatus,
)


TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS = 8.0


class LiveStateEvidenceFamily(StrEnum):
    CURRENT_TIME = "current_time"
    CURRENT_DATE = "current_date"
    SYSTEM_RESOURCES = "system_resources"
    SYSTEM_NETWORK = "system_network"
    SYSTEM_HARDWARE = "system_hardware"
    SYSTEM_SENSORS = "system_sensors"
    DAEMON_STATUS = "daemon_status"
    LIVE_STATE_MATH = "live_state_math"


@dataclass(frozen=True)
class LiveStateEvidencePlan:
    family: LiveStateEvidenceFamily | None
    evidence_required: bool
    candidate_tool_names: frozenset[str]
    missing_tool_names: frozenset[str]
    families: frozenset[LiveStateEvidenceFamily] = frozenset()
    missing_families: frozenset[LiveStateEvidenceFamily] = frozenset()
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class _LiveStateEvidenceDetection:
    families: tuple[LiveStateEvidenceFamily, ...]
    evidence_text: str
    evidence_text_by_family: dict[LiveStateEvidenceFamily, str]
    evidence_clauses_by_family: dict[LiveStateEvidenceFamily, tuple[str, ...]]


_ARITHMETIC_EXPRESSION_PATTERN = re.compile(
    r"(?i)(?:\d+(?:[.,]\d+)?|\be\b|\bpi\b|π)\s*(?:\*\*|[*/^×÷+\-])\s*"
    r"(?:\d+(?:[.,]\d+)?|\be\b|\bpi\b|π)"
)
_ARITHMETIC_TOKEN_PATTERN = re.compile(
    r"(?i)\d+(?:[.,]\d+)?|π|\bpi\b|\be\b|\*\*|[+\-*/^×÷()]"
)
_LIVE_STATE_THRESHOLD_PATTERN = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:"
    r"\b(?:over|above|under|below|greater\s+than|less\s+than|at\s+least|"
    r"at\s+most|more\s+than|not\s+more\s+than|higher\s+than|lower\s+than|"
    r"between|equal\s+to)\b|"
    r"[<>]=?|"
    r"больше|меньше|выше|ниже|превыш|как\s+минимум|как\s+максимум|"
    r"не\s+менее|не\s+более"
    r")"
    r".{0,40}"
    r"(?:\d+(?:[.,]\d+)?\s*(?:%|gb|mb|kb|gib|mib|ки?б|ми?б|ги?б)?)|"
    r"(?:\d+(?:[.,]\d+)?\s*(?:%|gb|mb|kb|gib|mib|ки?б|ми?б|ги?б)?)"
    r".{0,20}"
    r"(?:\b(?:or\s+higher|or\s+lower|or\s+less|or\s+more|or\s+above|or\s+below)\b|\+)"
    r")"
)
_LEGACY_LIVE_STATE_INTENT_PATTERN = re.compile(
    r"(?ix)"
    r"(?:"
    r"\bcpu\b|\bprocessor\b|\bmemory\b|\bram\b|\bload\b|\busage\b|"
    r"\bbattery\b|\bvpn\b|\bnetwork\b|\bexternal\s+ip\b|\bpublic\s+ip\b|"
    r"\bdaemon\b|\bsystem\s+status\b|\bhardware\b|\bdisk\b|"
    r"\bcurrent\s+(?:time|date)\b|\blocal\s+(?:time|date)\b|"
    r"\bwhat\s+time\s+is\s+it\b|"
    r"процессор|нагрузк|памят|оператив|батар|аккумулятор|сеть|"
    r"внешн\w*\s+(?:ip|айпи)|публичн\w*\s+(?:ip|айпи)|"
    r"\bцп\b|vpn|впн|желез|оборудован|диск|"
    r"сколько\s+врем(?:я|ени)|который\s+час|текущ\w*\s+(?:время|дата)"
    r")"
)
_LIVE_STATE_CONTEXT_EDGE_CHARS = " \t\r\n?!.:,;\"'`¿¡؟،。！？"
_CURRENT_TIME_TOOL_NAMES = frozenset({"datetime.now"})
_CURRENT_DATE_TOOL_NAMES = frozenset({"datetime.now"})
_DAEMON_STATUS_TOOL_NAMES = frozenset({"daemon.status"})
_SYSTEM_RESOURCE_TOOL_NAMES = frozenset({"tool.system.read.resources"})
_SYSTEM_NETWORK_TOOL_NAMES = frozenset({"tool.system.read.network"})
_SYSTEM_HARDWARE_TOOL_NAMES = frozenset({"tool.system.read.hardware"})
_SYSTEM_SENSOR_TOOL_NAMES = frozenset({"tool.system.read.sensors"})
_NON_RECOVERABLE_LIVE_STATE_ERROR_CODES = (
    NON_RECOVERABLE_TOOL_OBSERVATION_ERROR_CODES | frozenset({"invalid_arguments"})
)

_NON_LIVE_TIME_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"сколько\s+врем(?:я|ени)\s+(?:занимает|занял|займет|нужно|требуется)",
    r"время\s+выполнен|сложност\w*\s+.*врем|алгоритм",
    r"что\s+значит.*который\s+час",
    r"datetime\.now|time\.time|python",
    r"пример\w*|examples?",
    r"лог\w*|logs?|from\s+the\s+logs?",
    r"\bcron\b|таймер|напомни|\btimer\b|\bremind\s+me\b",
    r"\btime\s+complexity\b|\bexecution\s+time\b|\balgorithm\s+runtime\b",
    r"\bhow\s+(?:much\s+time|long\s+does)\b",
    r"\bwhat\s+does\b.*\bwhat\s+time\b.*\bmean\b",
)
_CURRENT_LOCAL_OBSERVATION_PATTERNS: tuple[str, ...] = (
    r"сейчас|в\s+данн\w*\s+момент",
    r"\b(?:right\s+now|now)\b",
)
_LIVE_STATE_CLAUSE_SEPARATOR_PATTERN = re.compile(
    r"[\r\n]+|[!?]+|(?<!\d)\.(?!\d)|(?<!\d),|,(?!\d)|[:;]+|"
    r"\b(?:and|also|then|but)\b|\b(?:и|а|но)\b",
    flags=re.IGNORECASE,
)
_NON_LIVE_DEFINITION_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bwhat\s+does\b.+\bmean\b",
    r"\bwhat(?:'s|\s+is)\b.+\b(?:concept|meaning)\b",
    r"\bwhat\s+is\s+(?:(?:a|an|the)\s+)?(?:network\s+interface|temperature\s+sensor|ip\s+address|cpu|processor|daemon|sensor|battery|disk|vpn)(?:\s+(?:right\s+)?now)?\s*$",
    r"\bwhat(?:'s|\s+is)\s+(?:cpu|processor|memory|ram|disk|battery|network|vpn)\s+(?:usage|load|utili[sz]ation)\s*$",
    r"\bhow\s+does\b.+\bwork\b",
    r"\bexplain\b.+\b(?:usage|load|network|daemon|sensor|battery|disk|vpn|ip)\b",
    r"что\s+(?:значит|такое)\b",
)
_NON_LIVE_OPERATIONAL_MEANING_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"как\s+работа(?:ет|ют)",
)
_NON_LIVE_MEANING_CONTEXT_PATTERNS: tuple[str, ...] = (
    *_NON_LIVE_DEFINITION_CONTEXT_PATTERNS,
    *_NON_LIVE_OPERATIONAL_MEANING_CONTEXT_PATTERNS,
)
_SHARED_SUFFIX_TOPIC_PATTERNS: tuple[str, ...] = (
    r"^(?:my\s+)?(?:vpn|wi[- ]?fi|wifi|internet|network|battery|disk|hardware|sensors?|temperature|thermal|daemon|впн|интернет|wi-fi|вайфай|сеть|батар\w*|аккумулятор|диск|желез\w*|оборудован\w*|датчик\w*|температур\w*|демон)$",
)
_SHARED_LIVE_SUFFIX_PATTERNS: tuple[str, ...] = (
    r"\bstatus\b",
    r"статус",
)
_NON_LIVE_SYSTEM_HISTORY_PATTERNS: tuple[str, ...] = (
    r"\b(?:from|in)\s+(?:the\s+)?logs?\b",
    r"\blogs?\b|\blog\s+history\b",
    r"\b(?:history|historical|yesterday|last\s+(?:night|week|month))\b",
    r"из\s+логов|в\s+логах|журнал\w*|истори\w*|вчера",
)
_HISTORY_COMPARISON_REQUIRES_CURRENT_PATTERNS: tuple[str, ...] = (
    r"\b(?:compare|compared|comparison)\b",
    r"\b(?:vs|versus)\b",
    r"\b(?:higher|lower|greater|less|more|above|below|over|under)\b.*\bthan\b",
    r"\b(?:than)\b.*\b(?:yesterday|last\s+(?:night|week|month)|history|historical|logs?)\b",
    r"(?:сравни|сравнен|больше|меньше|выше|ниже).*(?:вчера|истори|лог|журнал)",
)
_HISTORY_COMPARISON_CURRENT_MARKER_PATTERNS: tuple[str, ...] = (
    *_CURRENT_LOCAL_OBSERVATION_PATTERNS,
    r"\bcurrent\b",
    r"^\s*(?:is|are)\b(?!.*\b(?:historical|history|from\s+(?:the\s+)?logs?|in\s+(?:the\s+)?logs?)\b)",
    r"текущ",
)
_NON_LIVE_EXAMPLE_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\b(?:examples?|sample\s+code|code\s+example)\b",
    r"\b(?:write|generate|create|implement)\b.*\b(?:code|script|program|function)\b",
    r"\bpython\s+(?:code|script|program)\b",
    r"\bpython\s+snippet\b",
    r"\bgive\s+me\b.*\b(?:code|snippet)\b",
    r"\bgive\s+me\b.*\b(?:shell|python)?\s*script\b",
    r"\bshow\s+me\s+code\b",
    r"\bshow\s+me\b.*\bsnippet\b",
    r"\bshow\s+code\b",
    r"\bhow\s+can\s+i\s+check\b.+\bpython\b",
    r"\bhow\s+to\s+check\b.+\bpython\b",
)
_NON_LIVE_DURATION_OR_COMPLEXITY_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"сколько\s+врем(?:я|ени)\s+(?:занимает|занял|займет|нужно|требуется)",
    r"время\s+выполнен|сложност\w*\s+.*врем|алгоритм",
    r"\btime\s+complexity\b|\bexecution\s+time\b|\balgorithm\s+runtime\b",
    r"\bhow\s+(?:much\s+time|long\s+does|long\s+to)\b",
)
_NON_LIVE_SCHEDULING_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bcron\b|таймер|напомни|\btimer\b|\bremind\s+me\b",
)
_NON_LIVE_TASK_OR_EXAMPLE_CONTEXT_PATTERNS: tuple[str, ...] = (
    *_NON_LIVE_EXAMPLE_CONTEXT_PATTERNS,
    *_NON_LIVE_SCHEDULING_CONTEXT_PATTERNS,
)

_CURRENT_TIME_PATTERNS: tuple[str, ...] = (
    r"сколько\s+врем(?:я|ени)",
    r"который\s+час",
    r"сколько\s+сейчас\s+врем(?:я|ени)",
    r"сейчас\s+сколько\s+врем(?:я|ени)?",
    r"какое\s+время",
    r"какое\s+сейчас\s+время",
    r"какое\s+в\s+данн\w*\s+момент\s+время",
    r"какое\s+время\s+(?:сейчас|в\s+данн\w*\s+момент)",
    r"текущ\w*(?:\s+местн\w*)?\s+время",
    r"местн\w*\s+время",
    r"время\s+сейчас",
    r"(?:назови|скажи|подскажи)\s+время",
    r"сколько\s+на\s+часах",
    r"что\s+там\s+по\s+врем(?:я|ени)",
    r"через\s+сколько\s+врем",
    r"через\s+сколько\s+(?:секунд|минут|часов|дней)",
    r"(?:сколько|посчитай|прошло\s+сколько).*(?:секунд|минут|часов|дней).*(?:до\b|остал|прошл|пройдет|между|с\s+)",
    r"\bwhat\s+time\s+is\s+it\b",
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?time(?:\s+(?:right\s+)?now)?\b",
    r"\bcurrent\s+(?:local\s+)?time\b",
    r"\blocal\s+time\b",
    r"\btell\s+me\s+(?:what\s+time\s+it\s+is|the\s+time)\b",
    r"\bcan\s+you\s+(?:tell\s+me\s+the\s+time|give\s+me\s+the\s+current\s+time)\b",
    r"\b(?:the\s+)?time\s+please\b",
    r"\bgot\s+the\s+time\b",
    r"\bdo\s+you\s+know\s+what\s+time\s+it\s+is\b",
    r"\bwhat\s+does\s+the\s+clock\s+say\b",
    r"\btime\s+now\b",
    r"\bwhat\s+does\s+(?:datetime\.now|time\.time)\s+return\b.*\b(?:right\s+now|now)\b",
    r"\bhow\s+(?:long|many\s+(?:seconds?|minutes?|hours?|days?))\s+until\b",
    r"\bhow\s+many\s+(?:seconds?|minutes?|hours?|days?).*\b(?:since|from|between)\b",
    r"\b(?:seconds?|minutes?|hours?|days?)\s+since\b",
    r"\bcalculate\s+(?:seconds?|minutes?|hours?|days?)\s+(?:elapsed\s+)?since\b",
    r"(?:qué|que)\s+hora\s+es",
    r"hora\s+actual",
    r"quelle\s+heure",
    r"heure\s+actuelle",
    r"wie\s+spät\s+ist\s+es",
    r"aktuelle\s+uhrzeit",
    r"che\s+ore\s+sono",
    r"ora\s+attuale",
    r"que\s+horas\s+(?:são|sao)",
    r"hora\s+atual",
    r"котра\s+година",
    r"скільки\s+зараз\s+часу",
    r"kt[oó]ra\s+jest\s+godzina",
    r"aktualny\s+czas",
    r"saat\s+(?:kaç|kac)",
    r"şu\s+an\s+saat\s+(?:kaç|kac)",
    r"كم\s+الساعة",
    r"今何時",
    r"现在几点",
    r"現在幾點",
)
_CURRENT_TIME_DELTA_PATTERNS: tuple[str, ...] = (
    r"через\s+сколько\s+(?:секунд|минут|часов|дней|врем)",
    r"(?:сколько|посчитай|прошло\s+сколько).*(?:секунд|минут|часов|дней).*(?:до\b|остал|прошл|пройдет|между|с\s+)",
    r"\bhow\s+(?:long|many\s+(?:seconds?|minutes?|hours?|days?))\s+until\b",
    r"\bhow\s+many\s+(?:seconds?|minutes?|hours?|days?).*\b(?:since|from|between)\b",
    r"\b(?:seconds?|minutes?|hours?|days?)\s+since\b",
    r"\bcalculate\s+(?:seconds?|minutes?|hours?|days?)\s+(?:elapsed\s+)?since\b",
)
_CURRENT_TIME_UNTIL_SUPPORTED_TARGET_PATTERNS: tuple[str, ...] = (
    r"нов\w*\s+год",
    r"\bnew\s+year\b",
)
_CURRENT_DATE_PATTERNS: tuple[str, ...] = (
    r"какая\s+(?:сегодня\s+|сейчас\s+)?дата",
    r"какая\s+в\s+данн\w*\s+момент\s+дата",
    r"какая\s+дата\s+в\s+данн\w*\s+момент",
    r"какое\s+(?:сегодня\s+|сейчас\s+)?число",
    r"какой\s+(?:сегодня|сейчас)\s+день",
    r"какой\s+в\s+данн\w*\s+момент\s+день",
    r"какой\s+день\s+(?:сегодня|сейчас|в\s+данн\w*\s+момент)",
    r"сегодняшн\w*\s+дат",
    r"текущ\w*\s+дат",
    r"\bcurrent\s+date\b",
    r"\blocal\s+date\b",
    r"\btoday'?s\s+date\b",
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?date(?:\s+(?:right\s+)?now)?\b",
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?date\s+today\b",
    r"\bwhat\s+date\s+is\s+it\b",
    r"\bwhat\s+day\s+is\s+it(?:\s+today)?\b",
)
_SYSTEM_SENSOR_PATTERNS: tuple[str, ...] = (
    r"\b(?:current|cpu|system|thermal)\s+temperature\b",
    r"\btemperature\s+(?:now|reading|status|value)\b",
    r"\bthermal\s+(?:state|status|reading)\b",
    r"\bthermal\s+sensors?\s+(?:state|status|reading|value)\b",
    r"\bsensors?\s+(?:reading|status|value)\b",
    r"(?:сейчас|текущ|cpu|процессор|системн).*(?:температур|датчик)",
    r"(?:температур|датчик).*(?:cpu|процессор|системн|сейчас|текущ|статус|значен)",
)
_SYSTEM_RESOURCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:cpu|processor)\b.*\b(?:load|usage|used|utili[sz]ation|percent|%)\b",
    r"\b(?:cpu|processor|cores?)\b.*\bbusy\b",
    r"\b(?:load|usage|utili[sz]ation)\b.*\b(?:cpu|processor|memory|ram)\b",
    r"\b(?:memory|ram)\b.*\b(?:usage|used|free|available|current|utili[sz]ation)\b",
    r"\bdisk\b.*\b(?:usage|used|free|available|status|space)\b",
    r"\bdisk\s+space\b.*\b(?:available|free|used)\b",
    r"(?:свобод|занято|мест).*(?:диск|накопител)",
    r"(?:диск|накопител).*(?:свобод|занято|мест|статус)",
    r"\b(?:current|live)\s+(?:cpu|processor|memory|ram)\b",
    r"\b(?:cpu|processor|memory|ram)\b.*\b(?:over|above|under|below|greater|less|higher|lower)\b.*\d",
    r"\b(?:cpu|processor|memory|ram)\b.*\d+(?:[.,]\d+)?\s*(?:%|gb|mb|gib|mib)?\s*(?:\+|or\s+(?:higher|lower|less|more))",
    r"\b(?:current\s+)?system\s+load\b",
    r"\bload\s+average\b",
    r"\bsystem\s+load\b.*\b(?:over|above|under|below|greater|less|higher|lower)\b.*\d",
    r"нагрузк.*(?:cpu|процессор|цп|систем|сейчас|текущ)",
    r"(?:процессор|цп|систем).*(?:нагрузк)",
    r"(?:загрузк|загруж).*(?:cpu|процессор|цп)",
    r"(?:cpu|процессор|цп).*(?:загрузк|загруж)",
    r"(?:сейчас|текущ|сколько|занято|использ).*(?:процессор|цп|памят|оператив|ресурс)",
    r"(?:процессор|цп|памят|оператив|ресурс).*(?:сейчас|текущ|занято|использ|нагрузк)",
)
_SYSTEM_NETWORK_PATTERNS: tuple[str, ...] = (
    r"\bvpn\b.*\b(?:connected|status|current|up|down)\b",
    r"\bis\b.*\bvpn\b.*\b(?:connected|up|down|running)\b",
    r"\b(?:status|current|connected)\b.*\bvpn\b",
    r"\bnetwork\s+(?:status|interfaces?|sockets?|connection)\b",
    r"\bnetstat\b|\bnetwork\s+diagnostics\b",
    r"\blistening\s+on\s+port\s+\d+\b",
    r"\bam\s+i\s+online(?:\s+(?:right\s+)?now)?\b",
    r"\bam\s+i\s+connected\s+to\s+(?:the\s+)?internet\b",
    r"\b(?:wi[- ]?fi|wifi|internet)\b.*\b(?:connected|status|up|down|online)\b",
    r"\bis\s+(?:the\s+)?(?:wi[- ]?fi|wifi|internet)\s+(?:connected|up|down|online)\b",
    r"\b(?:vpn|internet|network|wi[- ]?fi|wifi)\b.*\bworking\b",
    r"\bis\s+(?:my\s+)?network\s+(?:up|down|connected|online|offline)\b",
    r"\bis\s+(?:my\s+)?(?:wi[- ]?fi|wifi)\s+(?:on|off)\b",
    r"\b(?:external|public|current|local|my)\s+ip(?:\s+address)?\b",
    r"(?:vpn|впн).*(?:подключ|статус|работ|включ|connected|status|up|down)",
    r"(?:подключ|статус|работ|включ).*(?:vpn|впн)",
    r"внешн\w*\s+(?:ip|айпи)|публичн\w*\s+(?:ip|айпи)",
    r"(?:интернет|wifi|wi-fi|вайфай).*(?:подключ|статус|работ|онлайн)",
    r"(?:работ).*(?:vpn|впн|интернет|wifi|wi-fi|вайфай|сеть)",
    r"(?:подключ|статус|работ|онлайн).*(?:интернет|wifi|wi-fi|вайфай)",
    r"(?:слуша|listen).*(?:порт|port)",
    r"(?:порт|port).*(?:слуша|listen)",
    r"(?:сеть|сетев).*(?:статус|подключ|сейчас|интерфейс|соедин)",
)
_SYSTEM_HARDWARE_PATTERNS: tuple[str, ...] = (
    r"\b(?:my|current|local)\s+(?:hardware|device)\b",
    r"\bwhat\s+(?:processor|cpu)\s+do\s+i\s+have\b",
    r"\bhow\s+much\s+(?:ram|memory)\s+do\s+i\s+have\b",
    r"\bhow\s+many\s+(?:cpu\s+)?cores?\b",
    r"\bcpu\s+core\s+count\b",
    r"\b(?:what\s+is\s+(?:the\s+)?)?(?:os|operating\s+system)\s+(?:version|build)\b",
    r"\b(?:which|what)\s+macos\s+(?:build|version)\b",
    r"\bhardware\s+(?:status|info|metadata)\b",
    r"(?:сколько|количеств).*(?:ядер|ядр).*(?:cpu|процессор|цп)",
    r"(?:cpu|процессор|цп).*(?:сколько|количеств|ядер|ядр)",
    r"\bbattery\b.*\b(?:left|charge|status|level|percent|percentage|remaining|current)\b",
    r"\bbattery\b.*\b(?:over|above|under|below|greater|less|higher|lower)\b.*\d",
    r"\b(?:how\s+much|current|remaining)\s+battery\b",
    r"\bos\s+metadata\b",
    r"какой\s+у\s+меня\s+(?:процессор|cpu|цп)",
    r"какой\s+(?:процессор|cpu|цп)\s+у\s+меня",
    r"какой\s+процессор.*(?:mac|мак|комп|ноут|устройств|этом)",
    r"(?:какой|какая|какую).*(?:macos|макос|ос|операцион).*(?:билд|сборк|верси)",
    r"(?:верси|билд|сборк).*(?:macos|макос|ос|операцион)",
    r"(?:мое|текущ|локальн).*(?:желез|оборудован|устройств)",
    r"(?:батар|аккумулятор).*(?:заряд|остал|сколько|сейчас|процент|уровень|статус)",
    r"(?:заряд|сколько|остал|процент|уровень|статус).*(?:батар|аккумулятор)",
    r"\bbattery\b.*(?:заряд|остал|сколько|процент|уровень|статус)",
    r"(?:заряд|сколько|остал|процент|уровень|статус).*\bbattery\b",
)
_SYSTEM_DISK_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\bdisk\b.*\b(?:usage|used|free|available|status|space)\b",
    r"\bdisk\s+space\b.*\b(?:available|free|used)\b",
    r"(?:свобод|занято|мест).*(?:диск|накопител)",
    r"(?:диск|накопител).*(?:свобод|занято|мест|статус)",
)
_SYSTEM_CPU_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:cpu|processor)\b.*\b(?:load|usage|used|utili[sz]ation|percent|%)\b",
    r"\b(?:cpu|processor|cores?)\b.*\bbusy\b",
    r"\b(?:current|live)\s+(?:cpu|processor)\b",
    r"\b(?:cpu|processor)\b.*\b(?:over|above|under|below|greater|less|higher|lower)\b.*\d",
    r"\b(?:cpu|processor)\b.*\d+(?:[.,]\d+)?\s*(?:%|gb|mb|gib|mib)?\s*(?:\+|or\s+(?:higher|lower|less|more))",
    r"(?:загрузк|загруж).*(?:cpu|процессор|цп)",
    r"(?:cpu|процессор|цп).*(?:загрузк|загруж)",
    r"(?:сейчас|текущ|сколько|занято|использ).*(?:процессор|цп)",
    r"(?:процессор|цп).*(?:сейчас|текущ|занято|использ|нагрузк)",
)
_SYSTEM_MEMORY_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:memory|ram)\b.*\b(?:usage|used|free|available|current|utili[sz]ation)\b",
    r"\b(?:usage|utili[sz]ation)\b.*\b(?:memory|ram)\b",
    r"\b(?:current|live)\s+(?:memory|ram)\b",
    r"\b(?:memory|ram)\b.*\b(?:over|above|under|below|greater|less|higher|lower)\b.*\d",
    r"\b(?:memory|ram)\b.*\d+(?:[.,]\d+)?\s*(?:%|gb|mb|gib|mib)?\s*(?:\+|or\s+(?:higher|lower|less|more))",
    r"(?:сейчас|текущ|сколько|занято|использ).*(?:памят|оператив)",
    r"(?:памят|оператив).*(?:сейчас|текущ|занято|использ)",
)
_SYSTEM_LOAD_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:cpu|processor)\b.*\b(?:load|usage|used|utili[sz]ation|percent|%)\b",
    r"\b(?:cpu|processor|cores?)\b.*\bbusy\b",
    r"\b(?:current\s+)?system\s+load\b",
    r"\bload\s+average\b",
    r"\bsystem\s+load\b.*\b(?:over|above|under|below|greater|less|higher|lower)\b.*\d",
    r"нагрузк.*(?:cpu|процессор|цп|систем|сейчас|текущ)",
    r"(?:процессор|цп|систем).*(?:нагрузк)",
)
_SYSTEM_LOAD_AVERAGE_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:current\s+)?system\s+load\b",
    r"\bload\s+average\b",
    r"\bsystem\s+load\b",
)
_SYSTEM_BATTERY_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\bbattery\b.*\b(?:left|charge|status|level|percent|percentage|remaining|current)\b",
    r"\bbattery\b.*\b(?:over|above|under|below|greater|less|higher|lower)\b.*\d",
    r"\b(?:how\s+much|current|remaining)\s+battery\b",
    r"(?:батар|аккумулятор).*(?:заряд|остал|сколько|сейчас|процент|уровень|статус)",
    r"(?:заряд|сколько|остал|процент|уровень|статус).*(?:батар|аккумулятор)",
)
_SYSTEM_OS_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:what\s+is\s+(?:the\s+)?)?(?:os|operating\s+system)\s+(?:version|build)\b",
    r"\b(?:which|what)\s+macos\s+(?:build|version)\b",
    r"\bos\s+metadata\b",
    r"(?:какой|какая|какую).*(?:macos|макос|ос|операцион).*(?:билд|сборк|верси)",
    r"(?:верси|билд|сборк).*(?:macos|макос|ос|операцион)",
)
_SYSTEM_HARDWARE_MEMORY_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\bhow\s+much\s+(?:ram|memory)\s+do\s+i\s+have\b",
    r"\b(?:installed|total|physical)\s+(?:ram|memory)\b",
    r"\b(?:ram|memory)\b.*\b(?:installed|total|capacity|do\s+i\s+have)\b",
    r"(?:сколько|объем|размер).*(?:памят|оператив)",
    r"(?:памят|оператив).*(?:сколько|объем|размер|у\s+меня)",
)
_SYSTEM_HARDWARE_CPU_CORE_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\bhow\s+many\s+(?:cpu\s+)?cores?\b",
    r"\bcpu\s+core\s+count\b",
    r"\b(?:logical|physical)\s+cpu\s+count\b",
    r"(?:сколько|количеств).*(?:ядер|ядр).*(?:cpu|процессор|цп)",
    r"(?:cpu|процессор|цп).*(?:сколько|количеств|ядер|ядр)",
)
_SYSTEM_HARDWARE_CPU_BRAND_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\bwhat\s+(?:processor|cpu)\s+do\s+i\s+have\b",
    r"\b(?:what|which)\s+(?:processor|cpu)\b",
    r"\b(?:processor|cpu)\s+(?:model|brand|string|name)\b",
    r"какой\s+у\s+меня\s+(?:процессор|cpu|цп)",
    r"какой\s+(?:процессор|cpu|цп)\s+у\s+меня",
    r"какой\s+процессор.*(?:mac|мак|комп|ноут|устройств|этом)",
)
_SYSTEM_VPN_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\bvpn\b.*\b(?:connected|status|current|up|down|running)\b",
    r"\bis\b.*\bvpn\b.*\b(?:connected|up|down|running)\b",
    r"\b(?:status|current|connected)\b.*\bvpn\b",
    r"(?:vpn|впн).*(?:подключ|статус|работ|включ|connected|status|up|down)",
    r"(?:подключ|статус|работ|включ).*(?:vpn|впн)",
)
_SYSTEM_PUBLIC_IP_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:external|public)\s+ip(?:\s+address)?\b",
    r"внешн\w*\s+(?:ip|айпи)|публичн\w*\s+(?:ip|айпи)",
)
_SYSTEM_LOCAL_IP_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:current|local|my)\s+ip(?:\s+address)?\b",
    r"\bip\s+address\b.*\b(?:current|local|my)\b",
    r"(?:локальн|текущ|мой|мо[её]).*(?:ip|айпи)",
)
_SYSTEM_CONNECTIVITY_EVIDENCE_PATTERNS: tuple[str, ...] = (
    r"\bam\s+i\s+online(?:\s+(?:right\s+)?now)?\b",
    r"\bam\s+i\s+connected\s+to\s+(?:the\s+)?internet\b",
    r"\b(?:wi[- ]?fi|wifi|internet)\b.*\b(?:connected|status|up|down|online)\b",
    r"\bis\s+(?:the\s+)?(?:wi[- ]?fi|wifi|internet)\s+(?:connected|up|down|online)\b",
    r"\b(?:internet|network|wi[- ]?fi|wifi)\b.*\bworking\b",
    r"\bis\s+(?:my\s+)?network\s+(?:up|down|connected|online|offline)\b",
    r"\bis\s+(?:my\s+)?(?:wi[- ]?fi|wifi)\s+(?:on|off)\b",
    r"(?:интернет|wifi|wi-fi|вайфай).*(?:подключ|статус|работ|онлайн)",
    r"(?:работ).*(?:интернет|wifi|wi-fi|вайфай|сеть)",
    r"(?:подключ|статус|работ|онлайн).*(?:интернет|wifi|wi-fi|вайфай)",
)
_DAEMON_STATUS_PATTERNS: tuple[str, ...] = (
    r"\bdaemon\s+(?:status|running|state)\b",
    r"\bdaemon\s+(?:health|healthy|unhealthy)\b",
    r"\bis\s+(?:the\s+)?daemon\s+(?:up|down|healthy|unhealthy|running)\b",
    r"\bstatus\s+of\s+(?:the\s+)?daemon\b",
    r"\b(?:system|runtime)\s+status\b",
    r"статус\s+демон|демон.*(?:статус|работ)",
)


def live_state_evidence_plan(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> LiveStateEvidencePlan:
    detection = _detect_live_state_evidence(request.user_input)
    scoped_text = detection.evidence_text
    base_families = detection.families
    if not base_families:
        return LiveStateEvidencePlan(
            family=None,
            evidence_required=False,
            candidate_tool_names=frozenset(),
            missing_tool_names=frozenset(),
        )

    family = base_families[0]
    families = frozenset(base_families)
    if (
        contains_arithmetic_expression(scoped_text)
        and _needs_calculator_for_live_state_math(scoped_text)
        and families
        & {
            LiveStateEvidenceFamily.CURRENT_TIME,
            LiveStateEvidenceFamily.CURRENT_DATE,
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            LiveStateEvidenceFamily.SYSTEM_SENSORS,
            LiveStateEvidenceFamily.DAEMON_STATUS,
        }
    ):
        family = LiveStateEvidenceFamily.LIVE_STATE_MATH
        families = frozenset({*families, LiveStateEvidenceFamily.LIVE_STATE_MATH})

    per_family_clause_candidates = _per_family_clause_candidates(
        base_families,
        request_plan,
        detection,
        scoped_text,
    )
    per_family_candidates = {
        base_family: frozenset(
            tool_name
            for _clause_text, clause_candidates in clause_candidates_by_text
            for tool_name in clause_candidates
        )
        for base_family, clause_candidates_by_text in per_family_clause_candidates.items()
    }
    candidate_tool_names = frozenset(
        tool_name
        for family_candidates in per_family_candidates.values()
        for tool_name in family_candidates
    )
    unavailable = any(
        not clause_candidates
        for clause_candidates_by_text in per_family_clause_candidates.values()
        for _clause_text, clause_candidates in clause_candidates_by_text
    )
    if family is LiveStateEvidenceFamily.LIVE_STATE_MATH:
        allowed = request_plan.allowed_tool_names or frozenset()
        if candidate_tool_names and "calculator.evaluate" in allowed:
            candidate_tool_names = frozenset({*candidate_tool_names, "calculator.evaluate"})
        elif candidate_tool_names:
            unavailable = True
        else:
            unavailable = True

    missing_tool_names = _missing_tool_names_for_family_candidates(
        per_family_clause_candidates,
        request,
        tool_observation_refs,
    )
    if family is LiveStateEvidenceFamily.LIVE_STATE_MATH and candidate_tool_names:
        calculator_missing = not _has_matching_completed_observation(
            "calculator.evaluate",
            request,
            tool_observation_refs,
            request_text=scoped_text,
        )
        if calculator_missing:
            missing_tool_names = frozenset({*missing_tool_names, "calculator.evaluate"})
    missing_families = _missing_families_for_family_candidates(
        per_family_candidates,
        missing_tool_names,
        family,
    )
    return LiveStateEvidencePlan(
        family=family,
        evidence_required=True,
        candidate_tool_names=candidate_tool_names,
        missing_tool_names=missing_tool_names,
        families=families,
        missing_families=missing_families,
        unavailable_reason="live_state_tool_unavailable" if unavailable else None,
    )


def _per_family_clause_candidates(
    base_families: tuple[LiveStateEvidenceFamily, ...],
    request_plan: ToolRequestPlan,
    detection: _LiveStateEvidenceDetection,
    fallback_text: str,
) -> dict[LiveStateEvidenceFamily, tuple[tuple[str, frozenset[str]], ...]]:
    candidates_by_family: dict[LiveStateEvidenceFamily, tuple[tuple[str, frozenset[str]], ...]] = {}
    for base_family in base_families:
        clauses = detection.evidence_clauses_by_family.get(base_family)
        if not clauses:
            clauses = (detection.evidence_text_by_family.get(base_family, fallback_text),)
        candidates_by_family[base_family] = tuple(
            (
                clause,
                _candidate_tool_names_for_family(base_family, request_plan, clause),
            )
            for clause in clauses
        )
    return candidates_by_family


def normalize_live_state_text(value: str) -> str:
    lowered = value.lower().strip(_LIVE_STATE_CONTEXT_EDGE_CHARS)
    lowered = re.sub(r"(?<!\d),|,(?!\d)|[:;]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip(_LIVE_STATE_CONTEXT_EDGE_CHARS)


def detect_live_state_family(value: str) -> LiveStateEvidenceFamily | None:
    families = detect_live_state_families(value)
    return families[0] if families else None


def detect_live_state_families(value: str) -> tuple[LiveStateEvidenceFamily, ...]:
    return _detect_live_state_evidence(value).families


def _detect_live_state_evidence(value: str) -> _LiveStateEvidenceDetection:
    normalized = normalize_live_state_text(value)
    clauses = _live_state_candidate_clauses(value)
    has_near_miss_clause = any(_is_non_live_near_miss_context(clause) for clause in clauses)
    clause_detection = _detect_live_state_evidence_from_live_clauses(clauses)
    if has_near_miss_clause:
        return clause_detection
    if _is_non_live_near_miss_context(normalized):
        return _LiveStateEvidenceDetection(
            families=(),
            evidence_text=normalized,
            evidence_text_by_family={},
            evidence_clauses_by_family={},
        )
    if len(clauses) > 1 and clause_detection.families:
        return clause_detection
    families = _detect_live_state_families_from_text(normalized)
    return _LiveStateEvidenceDetection(
        families=families,
        evidence_text=normalized,
        evidence_text_by_family={family: normalized for family in families},
        evidence_clauses_by_family={family: (normalized,) for family in families},
    )


def _detect_live_state_families_from_text(
    value: str,
) -> tuple[LiveStateEvidenceFamily, ...]:
    families: list[LiveStateEvidenceFamily] = []
    if _matches_any(_SYSTEM_SENSOR_PATTERNS, value):
        families.append(LiveStateEvidenceFamily.SYSTEM_SENSORS)
    if _matches_any(_SYSTEM_NETWORK_PATTERNS, value):
        families.append(LiveStateEvidenceFamily.SYSTEM_NETWORK)
    if _matches_any(_SYSTEM_RESOURCE_PATTERNS, value):
        families.append(LiveStateEvidenceFamily.SYSTEM_RESOURCES)
    if _matches_any(_DAEMON_STATUS_PATTERNS, value):
        families.append(LiveStateEvidenceFamily.DAEMON_STATUS)
    if _matches_any(_SYSTEM_HARDWARE_PATTERNS, value):
        families.append(LiveStateEvidenceFamily.SYSTEM_HARDWARE)
    excluded_time_context = _is_excluded_non_live_time_context(value)
    if _matches_any(_CURRENT_DATE_PATTERNS, value) and not excluded_time_context:
        families.append(LiveStateEvidenceFamily.CURRENT_DATE)
    if _matches_any(_CURRENT_TIME_PATTERNS, value) and not excluded_time_context:
        families.append(LiveStateEvidenceFamily.CURRENT_TIME)
    return tuple(dict.fromkeys(families))


def _detect_live_state_evidence_from_live_clauses(
    clauses: tuple[str, ...],
) -> _LiveStateEvidenceDetection:
    families: list[LiveStateEvidenceFamily] = []
    evidence_clauses: list[str] = []
    evidence_text_by_family: dict[LiveStateEvidenceFamily, list[str]] = {}
    evidence_clauses_by_family: dict[LiveStateEvidenceFamily, list[str]] = {}
    previous_live_context: str | None = None
    previous_live_families: tuple[LiveStateEvidenceFamily, ...] = ()
    pending_shared_suffix_clauses: list[str] = []
    suppress_bare_metric_clauses = False
    suppress_task_continuation_clauses = False
    for clause in clauses:
        if _is_non_live_near_miss_context(clause):
            previous_live_context = None
            previous_live_families = ()
            pending_shared_suffix_clauses.clear()
            suppress_bare_metric_clauses = True
            suppress_task_continuation_clauses = _near_miss_suppresses_task_continuations(clause)
            continue
        clause_families = _detect_live_state_families_from_text(clause)
        clause_text = clause
        if clause_families:
            if suppress_bare_metric_clauses and _is_bare_metric_clause(clause):
                continue
            if suppress_task_continuation_clauses and _is_non_live_task_continuation_clause(clause):
                continue
        if clause_families:
            suppress_bare_metric_clauses = False
            suppress_task_continuation_clauses = False
        if not clause_families and previous_live_context is not None:
            if _needs_calculator_for_live_state_math(clause):
                evidence_clauses.append(clause)
                for family in previous_live_families:
                    evidence_text_by_family.setdefault(family, []).append(clause)
                    evidence_clauses_by_family.setdefault(family, []).append(clause)
                continue
            contextual_clause = f"{previous_live_context} {clause}"
            contextual_families = _detect_live_state_families_from_text(contextual_clause)
            if contextual_families:
                clause_families = contextual_families
                clause_text = contextual_clause
        if not clause_families:
            if _matches_any(_SHARED_SUFFIX_TOPIC_PATTERNS, clause):
                pending_shared_suffix_clauses.append(clause)
            continue
        if pending_shared_suffix_clauses:
            for pending_clause in pending_shared_suffix_clauses:
                shared_clause_families = _detect_shared_suffix_families(
                    pending_clause,
                    clause,
                )
                if not shared_clause_families:
                    continue
                shared_clause_text = _shared_suffix_clause_text(pending_clause, clause)
                families.extend(shared_clause_families)
                evidence_clauses.append(shared_clause_text)
                for family in shared_clause_families:
                    evidence_text_by_family.setdefault(family, []).append(shared_clause_text)
                    evidence_clauses_by_family.setdefault(family, []).append(shared_clause_text)
            pending_shared_suffix_clauses.clear()
        families.extend(clause_families)
        evidence_clauses.append(clause_text)
        for family in clause_families:
            evidence_text_by_family.setdefault(family, []).append(clause_text)
            evidence_clauses_by_family.setdefault(family, []).append(clause_text)
        previous_live_context = clause_text
        previous_live_families = clause_families
    return _LiveStateEvidenceDetection(
        families=tuple(dict.fromkeys(families)),
        evidence_text="; ".join(evidence_clauses),
        evidence_text_by_family={
            family: "; ".join(family_clauses)
            for family, family_clauses in evidence_text_by_family.items()
        },
        evidence_clauses_by_family={
            family: tuple(family_clauses)
            for family, family_clauses in evidence_clauses_by_family.items()
        },
    )


def _detect_shared_suffix_families(
    topic_clause: str,
    suffix_source_clause: str,
) -> tuple[LiveStateEvidenceFamily, ...]:
    shared_clause_text = _shared_suffix_clause_text(topic_clause, suffix_source_clause)
    if shared_clause_text == topic_clause:
        return ()
    return _detect_live_state_families_from_text(shared_clause_text)


def _shared_suffix_clause_text(topic_clause: str, suffix_source_clause: str) -> str:
    suffix = _shared_live_suffix(suffix_source_clause)
    if suffix is None:
        return topic_clause
    return f"{topic_clause} {suffix}"


def _shared_live_suffix(value: str) -> str | None:
    if not _matches_any(_SHARED_LIVE_SUFFIX_PATTERNS, value):
        return None
    if "status" in value:
        return "status"
    if "статус" in value:
        return "статус"
    return None


def _live_state_candidate_clauses(value: str) -> tuple[str, ...]:
    return tuple(
        clause
        for clause in (
            normalize_live_state_text(part)
            for part in _LIVE_STATE_CLAUSE_SEPARATOR_PATTERN.split(value)
        )
        if clause
    )


def _is_non_live_near_miss_context(value: str) -> bool:
    if _matches_any(_NON_LIVE_DEFINITION_CONTEXT_PATTERNS, value):
        return True
    if _matches_any(_NON_LIVE_OPERATIONAL_MEANING_CONTEXT_PATTERNS, value) and not _matches_any(
        _CURRENT_LOCAL_OBSERVATION_PATTERNS,
        value,
    ):
        return True
    if _matches_any(_NON_LIVE_SYSTEM_HISTORY_PATTERNS, value):
        return not _history_context_requires_current_observation(value)
    if _matches_any(_NON_LIVE_DURATION_OR_COMPLEXITY_CONTEXT_PATTERNS, value):
        return True
    if _matches_any(_NON_LIVE_EXAMPLE_CONTEXT_PATTERNS, value):
        return True
    return _matches_any(_NON_LIVE_SCHEDULING_CONTEXT_PATTERNS, value) and not _matches_any(
        _CURRENT_TIME_PATTERNS,
        value,
    )


def _is_bare_metric_clause(value: str) -> bool:
    if _matches_any(_BARE_METRIC_CLAUSE_PATTERNS, value):
        return not _matches_any(
            _BARE_METRIC_LIVE_MARKER_PATTERNS,
            value,
        )
    return _matches_any(_BARE_METRIC_CLAUSE_WITH_WEAK_SUFFIX_PATTERNS, value) and not _matches_any(
        _BARE_METRIC_LIVE_MARKER_PATTERNS,
        value,
    )


def _near_miss_suppresses_task_continuations(value: str) -> bool:
    if _matches_any(_NON_LIVE_SYSTEM_HISTORY_PATTERNS, value):
        return False
    return (
        _matches_any(_NON_LIVE_DURATION_OR_COMPLEXITY_CONTEXT_PATTERNS, value)
        or _matches_any(_NON_LIVE_EXAMPLE_CONTEXT_PATTERNS, value)
        or _matches_any(_NON_LIVE_SCHEDULING_CONTEXT_PATTERNS, value)
    )


def _is_non_live_task_continuation_clause(value: str) -> bool:
    return _matches_any(_NON_LIVE_TASK_CONTINUATION_CLAUSE_PATTERNS, value)


_BARE_METRIC_CLAUSE_PATTERNS: tuple[str, ...] = (
    r"^(?:(?:cpu|processor|memory|ram|disk|battery|network)\s+)*(?:usage|load|status|space|level|charge)$",
    r"^(?:cpu|processor|memory|ram|disk|battery|network)$",
)
_BARE_METRIC_CLAUSE_WITH_WEAK_SUFFIX_PATTERNS: tuple[str, ...] = (
    r"^(?:(?:cpu|processor|memory|ram|disk|battery|network)\s+)*(?:usage|load|status|space|level|charge)(?:\s+(?:now|right\s+now|later))?$",
    r"^(?:cpu|processor|memory|ram|disk|battery|network)(?:\s+(?:now|right\s+now|later))?$",
)
_BARE_METRIC_LIVE_MARKER_PATTERNS: tuple[str, ...] = (
    r"\b(?:current|status|connected|up|down|online|available|free|used)\b",
    r"сейчас|текущ|статус|подключ|свобод|занято",
)
_NON_LIVE_TASK_CONTINUATION_CLAUSE_PATTERNS: tuple[str, ...] = (
    r"^(?:current\s+)?(?:(?:cpu|processor|memory|ram|disk|battery|network)\s+)*(?:usage|load|status|space|level|charge)(?:\s+(?:now|right\s+now|later))?$",
    r"^(?:whether|if|when)\s+(?:current\s+)?(?:(?:cpu|processor|memory|ram|disk|battery|network)\s+)*(?:usage|load|status|space|level|charge)\b.*(?:[<>]=?|\b(?:above|below|over|under|greater|less|higher|lower)\b)",
    r"^(?:is|are)\s+(?:current\s+)?(?:(?:cpu|processor|memory|ram|disk|battery|network)\s+)*(?:usage|load|status|space|level|charge)\b.*(?:[<>]=?|\b(?:above|below|over|under|greater|less|higher|lower)\b)",
    r"^check\s+(?:current\s+)?(?:(?:cpu|processor|memory|ram|disk|battery|network)\s+)*(?:usage|load|status|space|level|charge)(?:\s+(?:now|right\s+now|later))?$",
)
_INDEPENDENT_LIVE_CLAUSE_START_PATTERNS: tuple[str, ...] = (
    r"^(?:what|which|how\s+much|how\s+many|show|check|is|are|am|can|does|do)\b",
    r"^(?:покажи|проверь|какой|какая|сколько|есть\s+ли)\b",
)


def _history_context_requires_current_observation(value: str) -> bool:
    return _matches_any(_HISTORY_COMPARISON_CURRENT_MARKER_PATTERNS, value) and _matches_any(
        _HISTORY_COMPARISON_REQUIRES_CURRENT_PATTERNS,
        value,
    )


def _matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _needs_calculator_for_live_state_math(value: str) -> bool:
    return contains_arithmetic_expression(value) or _LIVE_STATE_THRESHOLD_PATTERN.search(value) is not None


def _is_excluded_non_live_time_context(value: str) -> bool:
    return _matches_any(
        _NON_LIVE_TIME_CONTEXT_PATTERNS, value
    ) and not _matches_any(_CURRENT_LOCAL_OBSERVATION_PATTERNS, value)


def _candidate_tool_names_for_family(
    family: LiveStateEvidenceFamily,
    request_plan: ToolRequestPlan,
    request_text: str,
) -> frozenset[str]:
    if family is LiveStateEvidenceFamily.CURRENT_TIME and _matches_any(
        _CURRENT_TIME_DELTA_PATTERNS,
        request_text,
    ):
        delta_required = frozenset({"datetime.now"})
        if _matches_any(_CURRENT_TIME_UNTIL_SUPPORTED_TARGET_PATTERNS, request_text):
            delta_required = frozenset({"datetime.now", "datetime.until"})
        delta_candidates = _allowed_live_tool_names(delta_required, request_plan)
        if delta_candidates:
            return delta_candidates
    required = _required_tool_names_for_family(family)
    return _allowed_live_tool_names(required, request_plan)


def _allowed_live_tool_names(
    required: frozenset[str],
    request_plan: ToolRequestPlan,
) -> frozenset[str]:
    allowed = request_plan.allowed_tool_names or frozenset()
    live_state_names = request_plan.live_state_tool_names or frozenset()
    return frozenset(
        tool_name
        for tool_name in required
        if tool_name in allowed
        and (tool_name in live_state_names or is_live_state_tool_name(tool_name))
    )


def _missing_tool_names_for_family_candidates(
    per_family_clause_candidates: dict[
        LiveStateEvidenceFamily,
        tuple[tuple[str, frozenset[str]], ...],
    ],
    request: LoopExecutionRequest,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> frozenset[str]:
    missing: set[str] = set()
    for family, clause_candidates_by_text in per_family_clause_candidates.items():
        for request_text, family_candidates in clause_candidates_by_text:
            if (
                family is LiveStateEvidenceFamily.CURRENT_TIME
                and _matches_any(_CURRENT_TIME_DELTA_PATTERNS, request_text)
                and _matches_any(_CURRENT_TIME_UNTIL_SUPPORTED_TARGET_PATTERNS, request_text)
                and len(family_candidates) > 1
                and not _current_time_delta_has_unsupported_target(request_text)
            ):
                has_any_delta_observation = (
                    not _has_conflicting_datetime_until_observation(
                        request_text,
                        tool_observation_refs,
                    )
                    and any(
                        _has_matching_completed_observation_for_delta(
                            tool_name,
                            request,
                            tool_observation_refs,
                            request_text=request_text,
                            require_observed_datetime_source=True,
                        )
                        for tool_name in family_candidates
                    )
                )
                if not has_any_delta_observation:
                    missing.update(family_candidates)
                continue
            missing.update(
                tool_name
                for tool_name in family_candidates
                if not _has_matching_completed_observation_for_delta(
                    tool_name,
                    request,
                    tool_observation_refs,
                    request_text=request_text,
                )
            )
    return frozenset(missing)


def _missing_families_for_family_candidates(
    per_family_candidates: dict[LiveStateEvidenceFamily, frozenset[str]],
    missing_tool_names: frozenset[str],
    primary_family: LiveStateEvidenceFamily,
) -> frozenset[LiveStateEvidenceFamily]:
    missing = {
        family
        for family, family_candidates in per_family_candidates.items()
        if family_candidates & missing_tool_names
    }
    if "calculator.evaluate" in missing_tool_names:
        missing.add(LiveStateEvidenceFamily.LIVE_STATE_MATH)
    if not missing and missing_tool_names:
        missing.add(primary_family)
    return frozenset(missing)


def _current_time_delta_has_unsupported_target(value: str) -> bool:
    return _matches_any(
        _CURRENT_TIME_DELTA_PATTERNS,
        value,
    ) and not _all_delta_clauses_support_datetime_until(
        value,
    )


def _all_delta_clauses_support_datetime_until(value: str) -> bool:
    delta_clauses = tuple(
        clause
        for clause in _live_state_candidate_clauses(value)
        if _matches_any(_CURRENT_TIME_DELTA_PATTERNS, clause)
    )
    if not delta_clauses:
        delta_clauses = (value,)
    return all(
        _matches_any(_CURRENT_TIME_UNTIL_SUPPORTED_TARGET_PATTERNS, clause)
        for clause in delta_clauses
    )


def _has_matching_completed_observation_for_delta(
    tool_name: str,
    request: LoopExecutionRequest,
    tool_observation_refs: tuple[ToolObservationRef, ...],
    *,
    request_text: str | None = None,
    require_observed_datetime_source: bool = False,
) -> bool:
    if tool_name == "datetime.until":
        expected_arguments = _expected_datetime_until_arguments(
            request_text or request.user_input
        )
        if expected_arguments is None:
            return False
        return any(
            _datetime_until_observation_matches_request(
                ref,
                expected_arguments=expected_arguments,
                tool_observation_refs=tool_observation_refs,
                require_observed_datetime_source=require_observed_datetime_source,
            )
            for ref in tool_observation_refs
        )
    return _has_matching_completed_observation(
        tool_name,
        request,
        tool_observation_refs,
        request_text=request_text,
    )


def _datetime_until_observation_matches_request(
    ref: ToolObservationRef,
    *,
    expected_arguments: dict[str, str],
    tool_observation_refs: tuple[ToolObservationRef, ...],
    require_observed_datetime_source: bool,
) -> bool:
    if not is_completed_observation(ref) or ref.tool_name != "datetime.until":
        return False
    observed_arguments = _datetime_until_observed_arguments(ref)
    if (
        observed_arguments.get("target") != expected_arguments["target"]
        or observed_arguments.get("unit") != expected_arguments["unit"]
    ):
        return False
    from_iso_values = _datetime_until_from_iso_values(ref)
    if not from_iso_values:
        return True
    if not require_observed_datetime_source:
        return True
    observed_sources = _completed_datetime_now_iso_values(tool_observation_refs)
    return bool(observed_sources) and from_iso_values <= observed_sources


def _has_conflicting_datetime_until_observation(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    expected_arguments = _expected_datetime_until_arguments(request_text)
    if expected_arguments is None:
        return False
    observed_sources = _completed_datetime_now_iso_values(tool_observation_refs)
    for ref in tool_observation_refs:
        if not is_completed_observation(ref) or ref.tool_name != "datetime.until":
            continue
        observed_arguments = _datetime_until_observed_arguments(ref)
        if (
            observed_arguments.get("target") != expected_arguments["target"]
            or observed_arguments.get("unit") != expected_arguments["unit"]
        ):
            continue
        from_iso_values = _datetime_until_from_iso_values(ref)
        if from_iso_values and (not observed_sources or not from_iso_values <= observed_sources):
            return True
    return False


def _datetime_until_observed_arguments(ref: ToolObservationRef) -> dict[str, str]:
    payload = _datetime_until_payload(ref)
    target = _first_string(ref.arguments.get("target"), payload.get("target"))
    unit = _first_string(ref.arguments.get("unit"), payload.get("unit"))
    if target is None or unit is None:
        return {}
    observed = {"target": target, "unit": unit}
    from_iso = _first_string(payload.get("from_iso"), ref.arguments.get("from_iso"))
    if from_iso is not None:
        observed["from_iso"] = from_iso
    return observed


def _datetime_until_payload(ref: ToolObservationRef) -> dict:
    payload = ref.structured_content
    if not isinstance(payload, dict) and ref.content_type == "application/json":
        try:
            parsed = json.loads(ref.content)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    return payload if isinstance(payload, dict) else {}


def _datetime_until_content_payload(ref: ToolObservationRef) -> dict:
    if ref.content_type != "application/json":
        return {}
    try:
        parsed = json.loads(ref.content)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _datetime_until_from_iso_values(ref: ToolObservationRef) -> frozenset[str]:
    values = {
        value
        for value in (
            ref.arguments.get("from_iso"),
            _datetime_until_payload(ref).get("from_iso"),
            _datetime_until_content_payload(ref).get("from_iso"),
        )
        if isinstance(value, str) and value.strip()
    }
    return frozenset(values)


def _first_string(*values: object) -> str | None:
    return next((value for value in values if isinstance(value, str)), None)


def _completed_datetime_now_iso_values(
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> frozenset[str]:
    values: set[str] = set()
    for ref in tool_observation_refs:
        if not is_completed_observation(ref) or ref.tool_name != "datetime.now":
            continue
        structured = ref.structured_content
        if isinstance(structured, dict):
            iso_value = structured.get("iso")
            if isinstance(iso_value, str) and iso_value.strip():
                values.add(iso_value)
                continue
        if ref.content_type != "application/json":
            continue
        try:
            parsed = json.loads(ref.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        iso_value = parsed.get("iso")
        if isinstance(iso_value, str) and iso_value.strip():
            values.add(iso_value)
    return frozenset(values)


def _expected_datetime_until_arguments(value: str) -> dict[str, str] | None:
    normalized = normalize_live_state_text(value)
    if not _matches_any(_CURRENT_TIME_UNTIL_SUPPORTED_TARGET_PATTERNS, normalized):
        return None
    unit = _expected_datetime_until_unit(normalized)
    if unit is None:
        return None
    return {"target": "next_new_year", "unit": unit}


def _expected_datetime_until_unit(value: str) -> str | None:
    if re.search(r"секунд|seconds?", value, flags=re.IGNORECASE):
        return "seconds"
    if re.search(r"минут|minutes?", value, flags=re.IGNORECASE):
        return "minutes"
    if re.search(r"часов|hours?", value, flags=re.IGNORECASE):
        return "hours"
    if re.search(r"дней|дня|days?", value, flags=re.IGNORECASE):
        return "days"
    return None


def _required_tool_names_for_family(family: LiveStateEvidenceFamily) -> frozenset[str]:
    if family is LiveStateEvidenceFamily.CURRENT_TIME:
        return _CURRENT_TIME_TOOL_NAMES
    if family is LiveStateEvidenceFamily.CURRENT_DATE:
        return _CURRENT_DATE_TOOL_NAMES
    if family is LiveStateEvidenceFamily.SYSTEM_RESOURCES:
        return _SYSTEM_RESOURCE_TOOL_NAMES
    if family is LiveStateEvidenceFamily.SYSTEM_NETWORK:
        return _SYSTEM_NETWORK_TOOL_NAMES
    if family is LiveStateEvidenceFamily.SYSTEM_HARDWARE:
        return _SYSTEM_HARDWARE_TOOL_NAMES
    if family is LiveStateEvidenceFamily.SYSTEM_SENSORS:
        return _SYSTEM_SENSOR_TOOL_NAMES
    if family is LiveStateEvidenceFamily.DAEMON_STATUS:
        return _DAEMON_STATUS_TOOL_NAMES
    return frozenset()


def _has_matching_completed_observation(
    tool_name: str,
    request: LoopExecutionRequest,
    tool_observation_refs: tuple[ToolObservationRef, ...],
    *,
    request_text: str | None = None,
) -> bool:
    if tool_name == "calculator.evaluate":
        return _has_all_matching_calculator_observations(
            request_text or request.user_input,
            tool_observation_refs,
        )
    if tool_name == "tool.system.read.resources":
        return _resource_observations_match_request(
            request_text or request.user_input,
            tool_observation_refs,
        )
    if tool_name == "tool.system.read.hardware":
        return _hardware_observations_match_request(
            request_text or request.user_input,
            tool_observation_refs,
        )
    if tool_name == "tool.system.read.network":
        return _network_observations_match_request(
            request_text or request.user_input,
            tool_observation_refs,
        )
    if tool_name.startswith("tool.system.read."):
        return any(
            is_completed_observation(ref)
            and ref.tool_name == tool_name
            and not _system_ref_is_unavailable(ref)
            for ref in tool_observation_refs
        )
    return any(
        is_completed_observation(ref) and ref.tool_name == tool_name
        for ref in tool_observation_refs
    )


def _resource_observations_match_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    refs = tuple(
        ref
        for ref in tool_observation_refs
        if is_completed_observation(ref) and ref.tool_name == "tool.system.read.resources"
    )
    if not refs:
        return False
    if _is_process_scoped_resource_request(request_text):
        return False
    requires_disk = _matches_any(_SYSTEM_DISK_EVIDENCE_PATTERNS, request_text)
    requires_cpu = _matches_any(_SYSTEM_CPU_EVIDENCE_PATTERNS, request_text)
    requires_memory = _matches_any(_SYSTEM_MEMORY_EVIDENCE_PATTERNS, request_text)
    requires_load_average = _matches_any(_SYSTEM_LOAD_AVERAGE_EVIDENCE_PATTERNS, request_text)
    requires_load = (
        _matches_any(_SYSTEM_LOAD_EVIDENCE_PATTERNS, request_text) and not requires_load_average
    )
    if not any((requires_disk, requires_cpu, requires_memory, requires_load, requires_load_average)):
        return any(_system_ref_can_satisfy_broad_family(ref) for ref in refs)
    if requires_disk and not any(_resource_ref_matches_disk(ref) for ref in refs):
        return False
    if requires_cpu and not any(_resource_ref_matches_cpu(ref) for ref in refs):
        return False
    if requires_memory and not any(_resource_ref_matches_memory(ref) for ref in refs):
        return False
    if requires_load and not any(_resource_ref_matches_load(ref) for ref in refs):
        return False
    if requires_load_average and not any(_resource_ref_matches_load_average(ref) for ref in refs):
        return False
    return True


def _resource_ref_matches_disk(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return _system_ref_has_schema(ref, "system.disk_free")
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv_command(ref) == "df"


def _resource_ref_matches_cpu(ref: ToolObservationRef) -> bool:
    if _resource_ref_matches_disk(ref):
        return False
    if _system_ref_has_schema(ref, "system.resource_overview") or _system_ref_has_schema(
        ref,
        "system.cpu_overview",
    ):
        return True
    if _system_ref_schema(ref) is not None:
        return False
    if _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv_command(ref) in {"top"}:
        return True
    return _resource_ref_is_generic_cpu_memory_snapshot(ref)


def _resource_ref_matches_memory(ref: ToolObservationRef) -> bool:
    if _resource_ref_matches_disk(ref):
        return False
    if _system_ref_has_schema(ref, "system.resource_overview") or _system_ref_has_schema(
        ref,
        "system.memory_overview",
    ):
        return True
    if _system_ref_schema(ref) is not None:
        return False
    if _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv_command(ref) in {
        "free",
        "vm_stat",
    }:
        return True
    return _resource_ref_is_generic_cpu_memory_snapshot(ref)


def _resource_ref_matches_load(ref: ToolObservationRef) -> bool:
    if _resource_ref_matches_disk(ref):
        return False
    if _system_ref_has_schema(ref, "system.resource_overview") or _system_ref_has_schema(
        ref,
        "system.cpu_overview",
    ):
        return True
    if _system_ref_schema(ref) is not None:
        return False
    if _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv_command(ref) in {
        "top",
        "uptime",
    }:
        return True
    return _resource_ref_is_generic_cpu_memory_snapshot(ref)


def _resource_ref_matches_load_average(ref: ToolObservationRef) -> bool:
    if _resource_ref_matches_disk(ref):
        return False
    if _system_ref_schema(ref) is not None:
        return (
            _system_ref_has_schema(ref, "system.resource_overview")
            or _system_ref_has_schema(ref, "system.cpu_overview")
        ) and _structured_has_any_key(
            ref,
            {"load_average", "load_1m", "load_5m", "load_15m"},
        )
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv_command(ref) == "uptime"


def _resource_ref_is_generic_cpu_memory_snapshot(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return False
    if not _system_ref_has_usable_raw_diagnostics(ref):
        return False
    argv = _system_ref_argv(ref)
    if argv and argv[0] not in {"top", "free", "vm_stat"}:
        return False
    metric = ref.arguments.get("metric")
    return not argv or metric in {"resources", "cpu_and_memory"}


def _hardware_observations_match_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    refs = tuple(
        ref
        for ref in tool_observation_refs
        if is_completed_observation(ref) and ref.tool_name == "tool.system.read.hardware"
    )
    if not refs:
        return False
    requires_battery = _matches_any(_SYSTEM_BATTERY_EVIDENCE_PATTERNS, request_text)
    requires_os = _matches_any(_SYSTEM_OS_EVIDENCE_PATTERNS, request_text)
    requires_memory = _matches_any(_SYSTEM_HARDWARE_MEMORY_EVIDENCE_PATTERNS, request_text)
    requires_core_count = _matches_any(_SYSTEM_HARDWARE_CPU_CORE_EVIDENCE_PATTERNS, request_text)
    requires_cpu_brand = _matches_any(_SYSTEM_HARDWARE_CPU_BRAND_EVIDENCE_PATTERNS, request_text)
    if not any(
        (
            requires_battery,
            requires_os,
            requires_memory,
            requires_core_count,
            requires_cpu_brand,
        )
    ):
        return any(_system_ref_can_satisfy_broad_family(ref) for ref in refs)
    if requires_battery and not any(_hardware_ref_matches_battery(ref) for ref in refs):
        return False
    if requires_os and not any(_hardware_ref_matches_os(ref) for ref in refs):
        return False
    if requires_memory and not any(_hardware_ref_matches_memory(ref) for ref in refs):
        return False
    if requires_core_count and not any(_hardware_ref_matches_core_count(ref) for ref in refs):
        return False
    if requires_cpu_brand and not any(_hardware_ref_matches_cpu_brand(ref) for ref in refs):
        return False
    return True


def _hardware_ref_matches_battery(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return _system_ref_has_schema(ref, "system.battery_charge")
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("pmset", "-g", "batt"),
        ("upower", "-i", "/org/freedesktop/UPower/devices/DisplayDevice"),
    }


def _hardware_ref_matches_os(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return _system_ref_has_schema(ref, "system.os_version")
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("sw_vers",),
        ("uname", "-a"),
    }


def _hardware_ref_matches_memory(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return _system_ref_has_schema(ref, "system.memory_overview")
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("sysctl", "-n", "hw.memsize"),
        ("lshw",),
        ("lshw", "-short"),
    }


def _hardware_ref_matches_core_count(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return _system_ref_has_schema(ref, "system.cpu_overview")
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("sysctl", "-n", "hw.ncpu"),
        ("sysctl", "-n", "hw.logicalcpu"),
        ("sysctl", "-n", "hw.physicalcpu"),
        ("lscpu",),
    }


def _hardware_ref_matches_cpu_brand(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return _system_ref_has_schema(ref, "system.cpu_overview") and _structured_has_any_key(
            ref,
            {"brand", "model", "processor", "processor_model", "cpu_model", "name"},
        )
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("sysctl", "-n", "machdep.cpu.brand_string"),
        ("lscpu",),
        ("lshw",),
        ("lshw", "-short"),
    }


def _network_observations_match_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    refs = tuple(
        ref
        for ref in tool_observation_refs
        if is_completed_observation(ref) and ref.tool_name == "tool.system.read.network"
    )
    if not refs:
        return False
    if _matches_any(_SYSTEM_VPN_EVIDENCE_PATTERNS, request_text):
        return any(_network_ref_matches_vpn(ref) for ref in refs)
    if _matches_any(_SYSTEM_PUBLIC_IP_EVIDENCE_PATTERNS, request_text):
        return any(_network_ref_matches_public_ip(ref) for ref in refs)
    if _matches_any(_SYSTEM_LOCAL_IP_EVIDENCE_PATTERNS, request_text):
        return any(_network_ref_matches_local_ip(ref) for ref in refs)
    if _matches_any(_SYSTEM_CONNECTIVITY_EVIDENCE_PATTERNS, request_text):
        return any(_network_ref_matches_connectivity(ref) for ref in refs)
    return any(_system_ref_can_satisfy_broad_family(ref) for ref in refs)


def _network_ref_matches_vpn(ref: ToolObservationRef) -> bool:
    if _system_ref_schema(ref) is not None:
        return _system_ref_has_schema(ref, "system.vpn_status")
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("scutil", "--nc", "list"),
        ("ip", "addr"),
    }


def _network_ref_matches_public_ip(ref: ToolObservationRef) -> bool:
    return _system_ref_has_schema(ref, "system.public_ip_address")


def _network_ref_matches_local_ip(ref: ToolObservationRef) -> bool:
    if _system_ref_has_schema(ref, "system.local_ip_address") or _system_ref_has_schema(
        ref,
        "system.network_interfaces",
    ):
        return True
    if _system_ref_schema(ref) is not None and _system_ref_argv(ref) != ("ip", "addr"):
        return False
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("ifconfig",),
        ("ip", "addr"),
    }


def _network_ref_matches_connectivity(ref: ToolObservationRef) -> bool:
    if _system_ref_has_schema(ref, "system.network_connectivity") or _system_ref_has_schema(
        ref,
        "system.network_interfaces",
    ):
        return True
    if _system_ref_schema(ref) is not None and _system_ref_argv(ref) != ("ip", "addr"):
        return False
    return _system_ref_has_usable_raw_diagnostics(ref) and _system_ref_argv(ref) in {
        ("ifconfig",),
        ("ip", "addr"),
    }


def _is_process_scoped_resource_request(request_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:process|daemon|service|pid|python|node|java|postgres|redis)\b",
            request_text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:процесс|демон|служб|пид)\b",
            request_text,
            flags=re.IGNORECASE,
        )
    )


def _system_ref_argv(ref: ToolObservationRef) -> tuple[str, ...]:
    argv = ref.arguments.get("argv")
    if not isinstance(argv, (list, tuple)):
        return ()
    if not all(isinstance(arg, str) for arg in argv):
        return ()
    return tuple(argv)


def _system_ref_argv_command(ref: ToolObservationRef) -> str | None:
    argv = _system_ref_argv(ref)
    return argv[0] if argv else None


def _system_ref_has_usable_raw_diagnostics(ref: ToolObservationRef) -> bool:
    if _system_ref_is_unavailable(ref):
        return False
    exit_code = ref.metadata.get("exit_code")
    if isinstance(exit_code, int):
        return exit_code == 0
    content_exit_code = _system_ref_content_exit_code(ref)
    if content_exit_code is not None:
        return content_exit_code == 0
    return not _system_ref_argv(ref)


def _system_ref_content_exit_code(ref: ToolObservationRef) -> int | None:
    if ref.content_type != "application/json":
        return None
    try:
        content = json.loads(ref.content)
    except (TypeError, ValueError):
        return None
    if not isinstance(content, dict):
        return None
    exit_code = content.get("exit_code")
    return exit_code if isinstance(exit_code, int) else None


def _system_ref_is_unavailable(ref: ToolObservationRef) -> bool:
    return ref.metadata.get("unavailable") is True


def _system_ref_can_satisfy_broad_family(ref: ToolObservationRef) -> bool:
    if _system_ref_is_unavailable(ref):
        return False
    if _system_ref_schema(ref) is not None:
        if not isinstance(ref.structured_content, dict):
            return False
        return ref.parse_status in {None, ToolParseStatus.PARSED, ToolParseStatus.PARTIAL}
    return _system_ref_has_usable_raw_diagnostics(ref)


def _system_ref_has_schema(ref: ToolObservationRef, schema: str) -> bool:
    if _system_ref_is_unavailable(ref):
        return False
    if _system_ref_schema(ref) != schema:
        return False
    if not isinstance(ref.structured_content, dict):
        return False
    if ref.parse_status is None:
        return True
    return ref.parse_status in {ToolParseStatus.PARSED, ToolParseStatus.PARTIAL}


def _structured_has_any_key(ref: ToolObservationRef, keys: set[str]) -> bool:
    if not isinstance(ref.structured_content, dict):
        return False
    return any(key in ref.structured_content for key in keys)


def _system_ref_schema(ref: ToolObservationRef) -> str | None:
    if isinstance(ref.structured_schema, str) and ref.structured_schema:
        return ref.structured_schema
    if isinstance(ref.structured_content, dict):
        schema = ref.structured_content.get("schema")
        if isinstance(schema, str) and schema:
            return schema
    return None


def _has_all_matching_calculator_observations(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    expected_expressions = arithmetic_expression_candidates(request_text)
    if not expected_expressions:
        return False
    completed_expressions = {
        normalize_calculator_expression(ref.arguments["expression"])
        for ref in tool_observation_refs
        if is_completed_observation(ref)
        and ref.tool_name == "calculator.evaluate"
        and isinstance(ref.arguments.get("expression"), str)
    }
    return all(
        normalize_calculator_expression(expected) in completed_expressions
        for expected in expected_expressions
    )


def tool_proposal_model_call_timeout(
    budget: LoopBudget,
    *,
    completed_observations: int,
    request: LoopExecutionRequest | None = None,
    request_plan: ToolRequestPlan | None = None,
    initial_model_call_cap_seconds: float = TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS,
) -> float:
    if completed_observations > 0:
        return float(budget.max_model_call_seconds)
    if (
        request is not None
        and request_plan is not None
        and request_requires_initial_tool_evidence(request, request_plan)
    ):
        return float(budget.max_model_call_seconds)
    return min(float(budget.max_model_call_seconds), float(initial_model_call_cap_seconds))


def request_requires_initial_tool_evidence(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
) -> bool:
    allowed = request_plan.allowed_tool_names or frozenset()
    if not allowed:
        return False
    if request_plan.final_answer_requires_observation():
        return True
    return final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    ) is not None


def final_answer_missing_evidence_plan(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> LiveStateEvidencePlan | None:
    plan = _without_terminal_unavailable_observations(
        live_state_evidence_plan(
            request,
            request_plan,
            tool_observation_refs=tool_observation_refs,
        ),
        request.user_input,
        tool_observation_refs,
    )
    if not plan.evidence_required:
        return None
    if not plan.candidate_tool_names:
        return None
    if not plan.missing_tool_names:
        return None
    if not (plan.missing_tool_names & plan.candidate_tool_names):
        raise RuntimeError("required_tool_evidence_missing")
    return plan


def failed_observation_exhausts_missing_evidence(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    observation_ref: ToolObservationRef,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=tool_observation_refs,
    )
    if not plan.evidence_required:
        return False
    if not is_live_state_tool_name(observation_ref.tool_name):
        return False
    if observation_ref.tool_name not in plan.missing_tool_names:
        return False
    if not _is_terminal_unavailable_observation(observation_ref):
        return False
    non_math_families = plan.families - frozenset({LiveStateEvidenceFamily.LIVE_STATE_MATH})
    if len(non_math_families) > 1:
        return False
    if _system_tool_request_requires_multiple_subtypes(request.user_input, observation_ref.tool_name):
        return False
    remaining_missing = plan.missing_tool_names - frozenset({observation_ref.tool_name})
    if plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH:
        return remaining_missing <= frozenset({"calculator.evaluate"})
    return not remaining_missing


def _system_tool_request_requires_multiple_subtypes(request_text: str, tool_name: str) -> bool:
    if tool_name == "tool.system.read.resources":
        subtypes: set[str] = set()
        requires_load_average = _matches_any(_SYSTEM_LOAD_AVERAGE_EVIDENCE_PATTERNS, request_text)
        if _matches_any(_SYSTEM_DISK_EVIDENCE_PATTERNS, request_text):
            subtypes.add("disk")
        if _matches_any(_SYSTEM_MEMORY_EVIDENCE_PATTERNS, request_text):
            subtypes.add("memory")
        if requires_load_average:
            subtypes.add("load_average")
        if _matches_any(_SYSTEM_CPU_EVIDENCE_PATTERNS, request_text) or (
            _matches_any(_SYSTEM_LOAD_EVIDENCE_PATTERNS, request_text) and not requires_load_average
        ):
            subtypes.add("cpu_load")
        return len(subtypes) > 1
    if tool_name == "tool.system.read.hardware":
        subtypes = {
            name
            for name, patterns in (
                ("battery", _SYSTEM_BATTERY_EVIDENCE_PATTERNS),
                ("os", _SYSTEM_OS_EVIDENCE_PATTERNS),
                ("memory", _SYSTEM_HARDWARE_MEMORY_EVIDENCE_PATTERNS),
                ("core_count", _SYSTEM_HARDWARE_CPU_CORE_EVIDENCE_PATTERNS),
                ("cpu_brand", _SYSTEM_HARDWARE_CPU_BRAND_EVIDENCE_PATTERNS),
            )
            if _matches_any(patterns, request_text)
        }
        return len(subtypes) > 1
    if tool_name == "tool.system.read.network":
        subtypes = {
            name
            for name, patterns in (
                ("vpn", _SYSTEM_VPN_EVIDENCE_PATTERNS),
                ("public_ip", _SYSTEM_PUBLIC_IP_EVIDENCE_PATTERNS),
                ("local_ip", _SYSTEM_LOCAL_IP_EVIDENCE_PATTERNS),
                ("connectivity", _SYSTEM_CONNECTIVITY_EVIDENCE_PATTERNS),
            )
            if _matches_any(patterns, request_text)
        }
        return len(subtypes) > 1
    return False


def _without_terminal_unavailable_observations(
    plan: LiveStateEvidencePlan,
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> LiveStateEvidencePlan:
    if not plan.missing_tool_names:
        return plan
    terminal_tool_names = frozenset(
        ref.tool_name
        for ref in tool_observation_refs
        if ref.tool_name in plan.missing_tool_names
        and is_live_state_tool_name(ref.tool_name)
        and not _system_tool_request_requires_multiple_subtypes(request_text, ref.tool_name)
        and _is_terminal_unavailable_observation(ref)
    )
    if not terminal_tool_names:
        return plan
    missing_tool_names = plan.missing_tool_names - terminal_tool_names
    return replace(
        plan,
        missing_tool_names=missing_tool_names,
        missing_families=_missing_families_for_tool_names(missing_tool_names, plan),
    )


def _is_terminal_unavailable_observation(ref: ToolObservationRef) -> bool:
    if ref.status == ToolObservationStatus.DENIED:
        return ref.error_code in {None, "tool_error", "tool_failed"}
    if ref.status not in {ToolObservationStatus.FAILED, ToolObservationStatus.TIMEOUT}:
        return False
    return ref.error_code not in _NON_RECOVERABLE_LIVE_STATE_ERROR_CODES


def _missing_families_for_tool_names(
    missing_tool_names: frozenset[str],
    plan: LiveStateEvidencePlan,
) -> frozenset[LiveStateEvidenceFamily]:
    if not missing_tool_names:
        return frozenset()
    families: set[LiveStateEvidenceFamily] = set()
    if "calculator.evaluate" in missing_tool_names:
        families.add(LiveStateEvidenceFamily.LIVE_STATE_MATH)
    if "datetime.now" in missing_tool_names:
        datetime_families = plan.families & {
            LiveStateEvidenceFamily.CURRENT_TIME,
            LiveStateEvidenceFamily.CURRENT_DATE,
        }
        families.update(datetime_families or {LiveStateEvidenceFamily.CURRENT_TIME})
    if "datetime.until" in missing_tool_names:
        families.add(LiveStateEvidenceFamily.CURRENT_TIME)
    tool_family_by_name = {
        "tool.system.read.resources": LiveStateEvidenceFamily.SYSTEM_RESOURCES,
        "tool.system.read.network": LiveStateEvidenceFamily.SYSTEM_NETWORK,
        "tool.system.read.hardware": LiveStateEvidenceFamily.SYSTEM_HARDWARE,
        "tool.system.read.sensors": LiveStateEvidenceFamily.SYSTEM_SENSORS,
        "daemon.status": LiveStateEvidenceFamily.DAEMON_STATUS,
    }
    families.update(
        family
        for tool_name, family in tool_family_by_name.items()
        if tool_name in missing_tool_names
    )
    if not families and plan.family is not None:
        families.add(plan.family)
    return frozenset(family for family in families if family in plan.families or family is plan.family)


def final_answer_deferred_missing_evidence_plan(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    tool_observation_refs: list[ToolObservationRef],
    used_tool_calls: int,
) -> LiveStateEvidencePlan | None:
    plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=tuple(tool_observation_refs),
    )
    if plan is None:
        return None
    if used_tool_calls >= request.budget.max_tool_calls:
        raise RuntimeError("required_tool_evidence_missing")
    return plan


def should_defer_final_answer_for_calculator_evidence(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    tool_observation_refs: list[ToolObservationRef],
    used_tool_calls: int,
) -> bool:
    allowed = request_plan.allowed_tool_names or frozenset()
    if "calculator.evaluate" not in allowed:
        return False
    plan = final_answer_deferred_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=tool_observation_refs,
        used_tool_calls=used_tool_calls,
    )
    if plan is None:
        return False
    return plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH


def request_needs_live_state_math_evidence(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    if "calculator.evaluate" not in (request_plan.allowed_tool_names or frozenset()):
        return False
    if not contains_arithmetic_expression(request.user_input):
        return False
    evidence_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=tool_observation_refs,
    )
    return (
        evidence_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
        and bool(evidence_plan.missing_tool_names)
    )


def calculator_observation_matches_request(
    ref: ToolObservationRef,
    request: LoopExecutionRequest,
) -> bool:
    return calculator_observation_matches_text(ref, request.user_input)


def calculator_observation_matches_text(
    ref: ToolObservationRef,
    value: str,
) -> bool:
    expected_expressions = arithmetic_expression_candidates(value)
    if not expected_expressions:
        return False
    actual = ref.arguments.get("expression")
    if not isinstance(actual, str):
        return False
    actual_normalized = normalize_calculator_expression(actual)
    return any(
        actual_normalized == normalize_calculator_expression(expected)
        for expected in expected_expressions
    )


def expected_calculator_expression(value: str) -> str | None:
    candidates = arithmetic_expression_candidates(value)
    if not candidates:
        return None
    return max(candidates, key=len)


def arithmetic_expression_candidates(value: str) -> list[str]:
    matches = list(_ARITHMETIC_TOKEN_PATTERN.finditer(value))
    if not matches:
        return []
    groups: list[list[re.Match[str]]] = []
    current = [matches[0]]
    for match in matches[1:]:
        gap = value[current[-1].end() : match.start()]
        if gap.strip():
            groups.append(current)
            current = [match]
        else:
            current.append(match)
    groups.append(current)

    candidates: list[str] = []
    for group in groups:
        text = value[group[0].start() : group[-1].end()].strip()
        if is_arithmetic_expression_candidate(text):
            candidates.append(text)
    return candidates


def is_arithmetic_expression_candidate(value: str) -> bool:
    if not re.search(r"(?:\*\*|[+\-*/^×÷])", value):
        return False
    balance = 0
    for char in value:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        if balance < 0:
            return False
    return balance == 0


def normalize_calculator_expression(value: str) -> str:
    return (
        value.replace("×", "*")
        .replace("÷", "/")
        .replace("π", "pi")
        .replace(",", ".")
        .replace(" ", "")
        .lower()
    )


def contains_arithmetic_expression(value: str) -> bool:
    return _ARITHMETIC_EXPRESSION_PATTERN.search(value) is not None


def contains_live_state_intent(value: str) -> bool:
    return _LEGACY_LIVE_STATE_INTENT_PATTERN.search(value) is not None


def is_completed_observation(ref: ToolObservationRef) -> bool:
    return ref.status in {ToolObservationStatus.COMPLETED, ToolObservationStatus.COMPLETED.value}


def is_live_state_tool_name(tool_name: str) -> bool:
    return (
        tool_name in {"datetime.now", "datetime.until", "daemon.status"}
        or tool_name.startswith("tool.system.read.")
    )
