from __future__ import annotations

from pathlib import Path

import pytest

from assistant_core.content_retrieval.project_docs import (
    MarkdownChunker,
    ProjectDocsSourceScanner,
    build_content_citation,
    plan_deleted_source,
    plan_reingestion,
)
from assistant_core.domain.content_retrieval import (
    ContentChunkStatus,
    ContentCitation,
    ContentSourceStatus,
    ContentSourceType,
)


pytestmark = pytest.mark.unit


def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_project_docs_source_allowlist_matches_readme_docs_and_adrs(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# Readme\n")
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "docs/adr/ADR-001_decision.md", "# Decision\n")
    _write(tmp_path, "notes.md", "# Not allowlisted\n")
    _write(tmp_path, "docs/nested/deep.md", "# Not allowlisted\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    by_path = {candidate.relative_path.as_posix(): candidate for candidate in candidates}
    assert set(by_path) == {
        "README.md",
        "docs/guide.md",
        "docs/adr/ADR-001_decision.md",
    }
    assert by_path["README.md"].source_type is ContentSourceType.README
    assert by_path["docs/guide.md"].source_type is ContentSourceType.PROJECT_DOC
    assert by_path["docs/adr/ADR-001_decision.md"].source_type is ContentSourceType.ADR
    assert by_path["docs/guide.md"].content_hash.startswith("sha256:")


def test_secret_like_paths_are_not_ingested(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "docs/.env.md", "DATABASE_URL=postgres://user:pass@example/db\n")
    _write(tmp_path, "docs/secrets.md", "token: abc\n")
    _write(tmp_path, "docs/adr/API_TOKEN.md", "# Token\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_symlink_to_secret_like_path_is_not_ingested(tmp_path: Path) -> None:
    _write(tmp_path, ".env", "DATABASE_URL=postgres://user:pass@example/db\n")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").symlink_to(tmp_path / ".env")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert candidates == []


def test_symlinked_docs_directory_is_not_ingested(tmp_path: Path) -> None:
    private_notes = tmp_path / "private_notes"
    private_notes.mkdir()
    (private_notes / "guide.md").write_text("# Private\n", encoding="utf-8")
    (tmp_path / "docs").symlink_to(private_notes)

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert candidates == []


def test_symlinked_docs_directory_adr_child_is_not_ingested(tmp_path: Path) -> None:
    private_notes = tmp_path / "private_notes"
    (private_notes / "adr").mkdir(parents=True)
    (private_notes / "adr" / "ADR-001_private.md").write_text("# Private\n", encoding="utf-8")
    (tmp_path / "docs").symlink_to(private_notes)

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert candidates == []


def test_symlinked_adr_directory_is_not_ingested(tmp_path: Path) -> None:
    private_adr = tmp_path / "private_adr"
    private_adr.mkdir()
    (private_adr / "ADR-001_private.md").write_text("# Private\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "adr").symlink_to(private_adr)

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert candidates == []


def test_secret_like_paths_include_common_key_material_names(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "docs/id_rsa.md", "private key\n")
    _write(tmp_path, "docs/known_hosts.md", "host key\n")
    _write(tmp_path, "docs/certificate.pem.md", "cert\n")
    _write(tmp_path, "docs/client.key.md", "client key\n")
    _write(tmp_path, "docs/server.crt.md", "server cert\n")
    _write(tmp_path, "docs/signing.cert.md", "signing cert\n")
    _write(tmp_path, "docs/archive.p12.md", "pkcs12 bundle\n")
    _write(tmp_path, "docs/archive.pfx.md", "pfx bundle\n")
    _write(tmp_path, "docs/p12.md", "pkcs12 bundle\n")
    _write(tmp_path, "docs/client-pfx.md", "pfx bundle\n")
    _write(tmp_path, "docs/pfx_bundle.md", "pfx bundle\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_secret_like_content_is_not_ingested(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nsafe project docs\n")
    _write(tmp_path, "docs/setup.md", "# Setup\nOPENAI_API_KEY=sk-live-secret-value-1234567890\n")
    _write(tmp_path, "docs/install.md", "# Install\n-----BEGIN PRIVATE KEY-----\nsecret\n")
    _write(tmp_path, "docs/examples.md", "# Examples\npassword: hunter2\n")
    _write(tmp_path, "docs/headers.md", "# Headers\nAuthorization: Bearer live-token\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_secret_like_content_includes_quoted_assignment_keys(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nsafe project docs\n")
    _write(tmp_path, "docs/json.md", '# JSON\n"OPENAI_API_KEY": "sk-live-secret-value-1234567890"\n')
    _write(tmp_path, "docs/toml.md", '# TOML\n"password" = "hunter2"\n')

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_secret_like_content_includes_common_key_naming_variants(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nsafe project docs\n")
    _write(tmp_path, "docs/config-a.md", "# Config\nsecret-key = hunter2\n")
    _write(tmp_path, "docs/config-b.md", "# Config\nclientSecret: hunter2\n")
    _write(tmp_path, "docs/config-c.md", "# Config\naccessToken: hunter2\n")
    _write(tmp_path, "docs/config-d.md", "# Config\nSECRET_VALUE=hunter2\n")
    _write(tmp_path, "docs/config-e.md", "# Config\nTOKEN_VALUE=hunter2\n")
    _write(tmp_path, "docs/config-f.md", "# Config\nPASSWORD_VALUE=hunter2\n")
    _write(tmp_path, "docs/config-g.md", "# Config\nAPI_KEY_VALUE=hunter2\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_secret_like_content_includes_inline_object_assignments(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nsafe project docs\n")
    _write(tmp_path, "docs/json.md", '# JSON\n{"clientSecret": "hunter2"}\n')
    _write(tmp_path, "docs/toml.md", '# TOML\noauth = { accessToken = "hunter2" }\n')

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_secret_like_content_includes_markdown_decorated_assignments(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nsafe project docs\n")
    _write(tmp_path, "docs/heading.md", "# Config\n## clientSecret: hunter2\n")
    _write(tmp_path, "docs/quote.md", "# Config\n> accessToken: hunter2\n")
    _write(tmp_path, "docs/list.md", "# Config\n* secret-key = hunter2\n")
    _write(tmp_path, "docs/code-span.md", "# Config\n`clientSecret`: hunter2\n")
    _write(tmp_path, "docs/bold.md", "# Config\n**clientSecret**: hunter2\n")
    _write(tmp_path, "docs/bold-list.md", "# Config\n- **secret-key** = hunter2\n")
    _write(tmp_path, "docs/quote-list-bold.md", "# Config\n> - **password**: hunter2\n")
    _write(tmp_path, "docs/quote-list-code.md", "# Config\n> - `secret-key` = hunter2\n")
    _write(tmp_path, "docs/task-list.md", "# Config\n- [ ] clientSecret: hunter2\n")
    _write(tmp_path, "docs/compact-quote.md", "# Config\n>clientSecret: hunter2\n")
    _write(tmp_path, "docs/nested-quote.md", "# Config\n>> - [ ] clientSecret: hunter2\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_secret_like_content_allows_schema_fields_that_mention_tokens(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nsafe project docs\n")
    _write(
        tmp_path,
        "docs/config.md",
        "# Config\nmax_input_tokens: 12000\ntoken_estimate: int\nallow_secret_in_memory: false\n",
    )

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == [
        "docs/config.md",
        "docs/guide.md",
    ]


def test_secret_like_paths_include_separator_variants(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "docs/api-key.md", "api key\n")
    _write(tmp_path, "docs/private-key.md", "private key\n")
    _write(tmp_path, "docs/private.key.md", "private key\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_secret_like_paths_include_space_and_backslash_variants(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "docs/api key.md", "api key\n")
    _write(tmp_path, "docs/private\\key.md", "private key\n")

    candidates = ProjectDocsSourceScanner(project_root=tmp_path).scan()

    assert [candidate.relative_path.as_posix() for candidate in candidates] == ["docs/guide.md"]


def test_markdown_chunker_splits_by_headings() -> None:
    content = "\n".join(
        [
            "# Jarvis",
            "intro",
            "## Setup",
            "install",
            "## Usage",
            "run",
        ],
    )

    chunks = MarkdownChunker(max_chars=500).chunk(
        source_id="source-1",
        source_path=Path("docs/guide.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        content=content,
        content_hash="sha256:source",
    )

    assert [chunk.heading_path for chunk in chunks] == [
        ["Jarvis"],
        ["Jarvis", "Setup"],
        ["Jarvis", "Usage"],
    ]
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]
    assert {chunk.status for chunk in chunks} == {ContentChunkStatus.ACTIVE}


def test_markdown_chunker_splits_oversized_sections() -> None:
    content = "# Long\n" + "\n".join(f"line {index} " + ("x" * 20) for index in range(12))

    chunks = MarkdownChunker(max_chars=120).chunk(
        source_id="source-1",
        source_path=Path("docs/long.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        content=content,
        content_hash="sha256:long",
    )

    assert len(chunks) > 1
    assert all(chunk.heading_path == ["Long"] for chunk in chunks)
    assert all(len(chunk.content) <= 120 for chunk in chunks)


def test_chunk_preserves_heading_path() -> None:
    content = "\n".join(
        [
            "# Root",
            "root text",
            "## Child",
            "child text",
            "### Leaf",
            "leaf text",
        ],
    )

    chunks = MarkdownChunker(max_chars=500).chunk(
        source_id="source-1",
        source_path=Path("docs/tree.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        content=content,
        content_hash="sha256:tree",
    )

    assert chunks[-1].heading_path == ["Root", "Child", "Leaf"]


def test_markdown_chunker_ignores_headings_inside_fenced_code() -> None:
    content = "\n".join(
        [
            "# Root",
            "intro",
            "```",
            "# Not A Heading",
            "code",
            "```",
            "## Real",
            "body",
        ],
    )

    chunks = MarkdownChunker(max_chars=500).chunk(
        source_id="source-1",
        source_path=Path("docs/code.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        content=content,
        content_hash="sha256:code",
    )

    assert [chunk.heading_path for chunk in chunks] == [["Root"], ["Root", "Real"]]
    assert "# Not A Heading" in chunks[0].content


def test_markdown_chunker_closes_fence_only_with_matching_marker() -> None:
    content = "\n".join(
        [
            "# Root",
            "```",
            "~~~",
            "# Still Code",
            "```",
            "## Real",
            "body",
        ],
    )

    chunks = MarkdownChunker(max_chars=500).chunk(
        source_id="source-1",
        source_path=Path("docs/fences.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        content=content,
        content_hash="sha256:fences",
    )

    assert [chunk.heading_path for chunk in chunks] == [["Root"], ["Root", "Real"]]
    assert "# Still Code" in chunks[0].content


def test_markdown_chunker_does_not_close_fence_on_marker_prefix_with_text() -> None:
    content = "\n".join(
        [
            "# Root",
            "```python",
            "```not-a-close",
            "# Still Code",
            "```",
            "## Real",
            "body",
        ],
    )

    chunks = MarkdownChunker(max_chars=500).chunk(
        source_id="source-1",
        source_path=Path("docs/fences.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        content=content,
        content_hash="sha256:fences",
    )

    assert [chunk.heading_path for chunk in chunks] == [["Root"], ["Root", "Real"]]
    assert "# Still Code" in chunks[0].content


def test_chunk_preserves_line_range_when_possible() -> None:
    content = "\n".join(
        [
            "# Root",
            "intro",
            "## Setup",
            "line a",
            "line b",
            "## Usage",
            "line c",
        ],
    )

    chunks = MarkdownChunker(max_chars=500).chunk(
        source_id="source-1",
        source_path=Path("docs/line-ranges.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        content=content,
        content_hash="sha256:lines",
    )

    setup = next(chunk for chunk in chunks if chunk.heading_path == ["Root", "Setup"])
    assert setup.line_start == 3
    assert setup.line_end == 5
    assert setup.citation == ContentCitation(
        path=Path("docs/line-ranges.md"),
        line_start=3,
        line_end=5,
        heading_path=["Root", "Setup"],
    )


def test_citation_formats_path_and_line_range() -> None:
    assert (
        build_content_citation(Path("docs/guide.md"), line_start=4, line_end=9).format()
        == "docs/guide.md:4-9"
    )


def test_changed_source_marks_old_chunks_stale() -> None:
    transition = plan_reingestion(
        existing_content_hash="sha256:old",
        new_content_hash="sha256:new",
    )

    assert transition.source_status is ContentSourceStatus.ACTIVE
    assert transition.previous_chunk_status is ContentChunkStatus.STALE
    assert transition.reingest_required is True


def test_deleted_source_marks_chunks_deleted_or_stale() -> None:
    transition = plan_deleted_source()

    assert transition.source_status is ContentSourceStatus.DELETED
    assert transition.chunk_status in {ContentChunkStatus.DELETED, ContentChunkStatus.STALE}
