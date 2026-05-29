from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from assistant_core.domain.content_retrieval import (
    ContentChunk,
    ContentChunkStatus,
    ContentCitation,
    ContentIngestionResult,
    ContentSourceStatus,
    ContentSourceSyncResult,
    ContentSourceType,
    DeletedSourcePlan,
    ReingestionPlan,
)
from assistant_core.domain.sensitivity import Sensitivity


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})(?P<tail>.*)$")
_SECRET_PATH_MARKERS = {
    ".env",
    ".cer",
    ".cert",
    ".crt",
    ".key",
    ".p12",
    ".pfx",
    ".ssh",
    ".aws",
    ".azure",
    ".gcp",
    ".kube",
    "api_key",
    "apikey",
    "auth",
    "cert",
    "certificate",
    "credential",
    "credentials",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
    "password",
    "passwd",
    "p12",
    "pfx",
    "pkcs12",
    "pem",
    "private_key",
    "secret",
    "secrets",
    "token",
}
_PRIVATE_KEY_BLOCK_RE = re.compile(r"(?i)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_URI_CREDENTIAL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)"
    r"(?:^|[,{]\s*)"
    r"(?:export[ \t]+)?"
    r"(?P<key>[`*_\"']*[A-Z0-9_.\- \t]+?[`*_\"']*)"
    r"[ \t]*(?:=|:)[ \t]*(?P<value>[^\n#,{}]+)",
)
_SECRET_KEY_EXACT = {
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "aws_access_key_id",
    "aws_secret_access_key",
    "database_url",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secret_key",
    "secretkey",
    "token",
}
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_password",
    "_passwd",
    "_private_key",
    "_secret",
    "_token",
)
_SECRET_PLACEHOLDER_VALUES = {
    "0",
    "1",
    "changeme",
    "example",
    "false",
    "none",
    "null",
    "placeholder",
    "redacted",
    "true",
    "your_api_key",
    "your_key",
    "your_password",
    "your_secret",
    "your_token",
}


@dataclass(frozen=True)
class ProjectDocsSourceCandidate:
    source_id: str
    relative_path: Path
    absolute_path: Path
    source_type: ContentSourceType
    title: str
    content: str
    content_hash: str
    sensitivity: Sensitivity = Sensitivity.PROJECT


class ContentIngestionStore(Protocol):
    async def sync_source_chunks(
        self,
        candidate: ProjectDocsSourceCandidate,
        chunks: list[ContentChunk],
    ) -> ContentSourceSyncResult: ...

    async def mark_missing_sources_deleted(self, seen_paths: set[Path]) -> tuple[int, int]: ...


class ProjectDocsSourceScanner:
    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root

    def scan(self) -> list[ProjectDocsSourceCandidate]:
        candidates: list[ProjectDocsSourceCandidate] = []
        for relative_path, source_type in self._candidate_paths():
            if _is_secret_like_path(relative_path):
                continue
            absolute_path = self._project_root / relative_path
            if not absolute_path.is_file():
                continue
            if _is_disallowed_source_target(self._project_root, absolute_path):
                continue
            content = absolute_path.read_text(encoding="utf-8")
            if _contains_secret_like_content(content):
                continue
            candidates.append(
                ProjectDocsSourceCandidate(
                    source_id=project_docs_source_id(relative_path),
                    relative_path=relative_path,
                    absolute_path=absolute_path,
                    source_type=source_type,
                    title=_title_from_markdown(content, relative_path),
                    content=content,
                    content_hash=_content_hash(content),
                ),
            )
        return sorted(candidates, key=lambda candidate: candidate.relative_path.as_posix())

    def _candidate_paths(self) -> list[tuple[Path, ContentSourceType]]:
        paths: list[tuple[Path, ContentSourceType]] = []
        readme = self._project_root / "README.md"
        if readme.exists():
            paths.append((Path("README.md"), ContentSourceType.README))
        docs_dir = self._project_root / "docs"
        if not docs_dir.is_dir() or docs_dir.is_symlink():
            return paths
        if docs_dir.is_dir() and not docs_dir.is_symlink():
            paths.extend(
                (Path("docs") / path.name, ContentSourceType.PROJECT_DOC)
                for path in sorted(docs_dir.glob("*.md"))
            )
            adr_dir = docs_dir / "adr"
            if adr_dir.is_dir() and not adr_dir.is_symlink():
                paths.extend(
                    (Path("docs") / "adr" / path.name, ContentSourceType.ADR)
                    for path in sorted(adr_dir.glob("*.md"))
                )
        return paths


class MarkdownChunker:
    def __init__(self, *, max_chars: int = 1800) -> None:
        if max_chars < 80:
            raise ValueError("max_chars must be at least 80")
        self._max_chars = max_chars

    def chunk(
        self,
        *,
        source_id: str,
        source_path: Path,
        source_type: ContentSourceType,
        content: str,
        content_hash: str,
        sensitivity: Sensitivity = Sensitivity.PROJECT,
    ) -> list[ContentChunk]:
        sections = _markdown_sections(content)
        chunks: list[ContentChunk] = []
        ordinal = 0
        for section in sections:
            for split in _split_section(section, self._max_chars):
                citation = build_content_citation(
                    source_path,
                    line_start=split.line_start,
                    line_end=split.line_end,
                    heading_path=split.heading_path,
                )
                chunks.append(
                    ContentChunk(
                        chunk_id=_chunk_id(
                            source_id=source_id,
                            content_hash=content_hash,
                            ordinal=ordinal,
                            line_start=split.line_start,
                            line_end=split.line_end,
                        ),
                        source_id=source_id,
                        ordinal=ordinal,
                        source_path=source_path,
                        source_type=source_type,
                        heading_path=split.heading_path,
                        content=split.content,
                        content_hash=content_hash,
                        line_start=split.line_start,
                        line_end=split.line_end,
                        citation=citation,
                        sensitivity=sensitivity,
                        status=ContentChunkStatus.ACTIVE,
                        metadata={},
                    ),
                )
                ordinal += 1
        return chunks


class ProjectDocsIngestionService:
    def __init__(
        self,
        *,
        store: ContentIngestionStore,
        scanner: ProjectDocsSourceScanner,
        chunker: MarkdownChunker,
    ) -> None:
        self._store = store
        self._scanner = scanner
        self._chunker = chunker

    async def ingest(self) -> ContentIngestionResult:
        seen_paths: set[Path] = set()
        created_sources = 0
        updated_sources = 0
        created_chunks = 0
        stale_chunks = 0
        candidates = self._scanner.scan()

        for candidate in candidates:
            seen_paths.add(candidate.relative_path)
            chunks = self._chunker.chunk(
                source_id=candidate.source_id,
                source_path=candidate.relative_path,
                source_type=candidate.source_type,
                content=candidate.content,
                content_hash=candidate.content_hash,
                sensitivity=candidate.sensitivity,
            )
            sync_result = await self._store.sync_source_chunks(candidate, chunks)
            created_sources += 1 if sync_result.created_source else 0
            updated_sources += 1 if sync_result.updated_source else 0
            created_chunks += sync_result.created_chunks
            stale_chunks += sync_result.stale_chunks

        deleted_sources, deleted_chunks = await self._store.mark_missing_sources_deleted(seen_paths)
        return ContentIngestionResult(
            seen_sources=len(candidates),
            created_sources=created_sources,
            updated_sources=updated_sources,
            deleted_sources=deleted_sources,
            created_chunks=created_chunks,
            stale_chunks=stale_chunks,
            deleted_chunks=deleted_chunks,
        )


@dataclass(frozen=True)
class _MarkdownSection:
    heading_path: list[str]
    lines: list[tuple[int, str]]


@dataclass(frozen=True)
class _ChunkSplit:
    heading_path: list[str]
    content: str
    line_start: int
    line_end: int


def build_content_citation(
    path: Path,
    *,
    line_start: int,
    line_end: int,
    heading_path: list[str] | None = None,
) -> ContentCitation:
    return ContentCitation(
        path=path,
        line_start=line_start,
        line_end=line_end,
        heading_path=list(heading_path or []),
    )


def project_docs_source_id(path: Path) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-content-source:{path.as_posix()}"))


def plan_reingestion(
    *,
    existing_content_hash: str | None,
    new_content_hash: str,
) -> ReingestionPlan:
    changed = existing_content_hash is not None and existing_content_hash != new_content_hash
    return ReingestionPlan(
        reingest_required=existing_content_hash is None or changed,
        source_status=ContentSourceStatus.ACTIVE,
        previous_chunk_status=ContentChunkStatus.STALE if changed else None,
    )


def plan_deleted_source() -> DeletedSourcePlan:
    return DeletedSourcePlan(
        source_status=ContentSourceStatus.DELETED,
        chunk_status=ContentChunkStatus.DELETED,
    )


def _markdown_sections(content: str) -> list[_MarkdownSection]:
    lines = content.splitlines()
    if not lines:
        return []

    sections: list[_MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[tuple[int, str]] = []
    current_heading_path: list[str] = []
    fence_marker: str | None = None

    def flush() -> None:
        nonlocal current_lines, current_heading_path
        if current_lines and any(line.strip() for _, line in current_lines):
            sections.append(
                _MarkdownSection(
                    heading_path=list(current_heading_path),
                    lines=current_lines,
                ),
            )
        current_lines = []

    for line_number, line in enumerate(lines, start=1):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group("marker")
            if fence_marker is None:
                fence_marker = marker
                current_lines.append((line_number, line))
                continue
            if (
                marker[0] == fence_marker[0]
                and len(marker) >= len(fence_marker)
                and not fence_match.group("tail").strip()
            ):
                fence_marker = None
                current_lines.append((line_number, line))
                continue
        match = _HEADING_RE.match(line)
        if match and fence_marker is None:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [(item_level, item_title) for item_level, item_title in heading_stack if item_level < level]
            heading_stack.append((level, title))
            current_heading_path = [item_title for _, item_title in heading_stack]
        current_lines.append((line_number, line))

    flush()
    return sections


def _split_section(section: _MarkdownSection, max_chars: int) -> list[_ChunkSplit]:
    splits: list[_ChunkSplit] = []
    current: list[tuple[int, str]] = []

    def current_text(extra: str | None = None) -> str:
        lines = [line for _, line in current]
        if extra is not None:
            lines.append(extra)
        return "\n".join(lines).strip()

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = "\n".join(line for _, line in current).strip()
        if text:
            splits.append(
                _ChunkSplit(
                    heading_path=list(section.heading_path),
                    content=text,
                    line_start=current[0][0],
                    line_end=current[-1][0],
                ),
            )
        current = []

    for line_number, line in section.lines:
        if current and len(current_text(line)) > max_chars:
            flush()
        if len(line) > max_chars:
            flush()
            for offset in range(0, len(line), max_chars):
                part = line[offset : offset + max_chars].strip()
                if part:
                    splits.append(
                        _ChunkSplit(
                            heading_path=list(section.heading_path),
                            content=part,
                            line_start=line_number,
                            line_end=line_number,
                        ),
                    )
            continue
        current.append((line_number, line))
    flush()
    return splits


def _title_from_markdown(content: str, relative_path: Path) -> str:
    for line in content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    if relative_path.name == "README.md":
        return "README"
    return relative_path.stem.replace("_", " ").replace("-", " ").strip() or relative_path.name


def _content_hash(content: str) -> str:
    return "sha256:" + sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(
    *,
    source_id: str,
    content_hash: str,
    ordinal: int,
    line_start: int,
    line_end: int,
) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"jarvis-content-chunk:{source_id}:{content_hash}:{ordinal}:{line_start}:{line_end}",
        ),
    )


def _is_secret_like_path(relative_path: Path) -> bool:
    for part in relative_path.parts:
        lowered = part.lower()
        path_part = Path(lowered)
        stems: set[str] = set()
        stem = path_part.stem
        while stem and stem not in stems:
            stems.add(stem)
            stem = Path(stem).stem
        normalized = {
            lowered,
            _separator_normalized(lowered),
            _separator_compacted(lowered),
        }
        for stem_value in stems:
            normalized.add(stem_value)
            normalized.add(_separator_normalized(stem_value))
            normalized.add(_separator_compacted(stem_value))
        candidates = {lowered, *stems, *path_part.suffixes, *normalized}
        if any(
            marker in candidates
            or marker in lowered
            or marker in normalized
            or marker.replace("_", "") in normalized
            for marker in _SECRET_PATH_MARKERS
        ):
            return True
    return False


def _contains_secret_like_content(content: str) -> bool:
    if _PRIVATE_KEY_BLOCK_RE.search(content) or _URI_CREDENTIAL_RE.search(content):
        return True
    for line in content.splitlines():
        normalized_line = _strip_markdown_assignment_prefix(line)
        for match in _SECRET_ASSIGNMENT_RE.finditer(normalized_line):
            if not _is_secret_like_key(match.group("key")):
                continue
            if _looks_like_secret_value(match.group("value")):
                return True
    return False


def _strip_markdown_assignment_prefix(line: str) -> str:
    stripped = line.strip()
    previous = None
    while stripped != previous:
        previous = stripped
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = re.sub(r"^>+\s*", "", stripped)
        stripped = re.sub(r"^[-*+]\s+(?:\[[ xX]\]\s+)?", "", stripped)
    return stripped


def _is_secret_like_key(key: str) -> bool:
    raw_key = key.strip().strip("`*_ '\"").strip()
    normalized = _separator_normalized(_camel_tokenized(raw_key).lower()).strip("_")
    compacted = _separator_compacted(normalized)
    if normalized in _SECRET_KEY_EXACT or compacted in _SECRET_KEY_EXACT:
        return True
    if any(normalized.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES):
        return True
    if normalized.endswith("_value"):
        root = normalized.removesuffix("_value")
        compacted_root = _separator_compacted(root)
        return root in _SECRET_KEY_EXACT or compacted_root in _SECRET_KEY_EXACT or any(
            root.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES
        )
    return False


def _camel_tokenized(value: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)


def _looks_like_secret_value(value: str) -> bool:
    normalized = value.strip().strip("`'\"").strip()
    lowered = normalized.lower()
    if not normalized or lowered in _SECRET_PLACEHOLDER_VALUES:
        return False
    if lowered.startswith(("your_", "your-", "<")) or lowered.endswith(">"):
        return False
    return bool(re.search(r"[a-zA-Z0-9]", normalized))


def _separator_normalized(value: str) -> str:
    return re.sub(r"[\s\\._-]+", "_", value)


def _separator_compacted(value: str) -> str:
    return re.sub(r"[\s\\._-]+", "", value)


def _is_disallowed_source_target(project_root: Path, absolute_path: Path) -> bool:
    if absolute_path.is_symlink():
        return True
    try:
        resolved_root = project_root.resolve(strict=True)
        resolved_path = absolute_path.resolve(strict=True)
    except OSError:
        return True
    try:
        relative_resolved_path = resolved_path.relative_to(resolved_root)
    except ValueError:
        return True
    return _is_secret_like_path(relative_resolved_path)
