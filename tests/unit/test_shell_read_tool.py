from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_core.tools.shell_read import ShellCommandClassifier
from assistant_core.tools.shell_read import SubprocessShellExecutor


pytestmark = pytest.mark.unit


def _classifier(root: Path) -> ShellCommandClassifier:
    return ShellCommandClassifier(allowed_roots=[root], max_lines=120)


def test_allows_pwd_inside_workspace(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["pwd"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "pwd"


def test_allows_ls_inside_workspace(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["ls", "."], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "ls"


def test_allows_bare_ls_at_workspace_root_with_git_metadata(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["ls"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "ls"


def test_denies_ls_when_requested_listing_exposes_secret_like_entry(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    decision = _classifier(tmp_path).classify(["ls", "-a", "."], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"


@pytest.mark.parametrize("argv", [["ls", "-l", "."], ["ls", "-la", "."]])
def test_allows_safe_ls_flags_inside_workspace(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "ls"


def test_allows_rg_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("needle\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["rg", "needle", "notes.md"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "rg"


def test_allows_sed_n_with_bounded_range(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("a\nb\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(
        ["sed", "-n", "1,20p", "notes.md"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == "sed"


def test_allows_head_and_tail_with_bounded_line_count(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("a\nb\n", encoding="utf-8")
    classifier = _classifier(tmp_path)

    head = classifier.classify(["head", "-n", "20", "notes.md"], cwd=tmp_path)
    tail = classifier.classify(["tail", "-n", "20", "notes.md"], cwd=tmp_path)

    assert head.allowed is True
    assert head.family == "head"
    assert tail.allowed is True
    assert tail.family == "tail"


def test_allows_wc_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("a\nb\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["wc", "-l", "notes.md"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "wc"


def test_allows_git_status_short_with_explicit_safe_pathspec(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("hello\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(
        ["git", "status", "--short", "--", "docs/notes.md"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == "git.status"


def test_allows_git_diff_read_only(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("hello\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["git", "diff", "--", "docs/notes.md"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "git.diff"


def test_allows_git_show_read_only(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("hello\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(
        ["git", "show", "--stat", "--no-patch", "HEAD", "--", "docs/notes.md"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == "git.show"


@pytest.mark.parametrize(
    "argv",
    [
        ["rg", "needle", ".", "|", "head"],
        ["rg", "needle; rm -rf ."],
        ["sh", "-c", "pwd"],
        ["FOO=bar", "pwd"],
    ],
)
def test_denies_shell_metacharacters(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "shell_syntax_denied"


@pytest.mark.parametrize("argv", [["./rg", "needle", "docs"], ["/tmp/rg", "needle", "docs"]])
def test_denies_command_paths_even_with_allowlisted_basename(
    tmp_path: Path,
    argv: list[str],
) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "command_path_denied"


@pytest.mark.parametrize("argv", [["ls", "-R", "."], ["ls", "-L", "."], ["ls", "-LR", "."]])
def test_denies_ls_recursive_or_follow_symlink_flags(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "unsupported_arguments"


def test_denies_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "notes.md").write_text("secret\n", encoding="utf-8")

    decision = _classifier(workspace).classify(
        ["head", "-n", "5", "../outside/notes.md"],
        cwd=workspace,
    )

    assert decision.allowed is False
    assert decision.code == "path_outside_workspace"


def test_denies_symlink_escape_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "notes.md").write_text("secret\n", encoding="utf-8")
    (workspace / "linked.md").symlink_to(outside / "notes.md")

    decision = _classifier(workspace).classify(["head", "-n", "5", "linked.md"], cwd=workspace)

    assert decision.allowed is False
    assert decision.code == "path_outside_workspace"


def test_denies_directory_with_symlink_descendant_to_secret_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    docs = workspace / "docs"
    workspace.mkdir()
    outside.mkdir()
    docs.mkdir()
    (outside / "id_rsa").write_text("secret\n", encoding="utf-8")
    (docs / "current").symlink_to(outside / "id_rsa")

    decision = _classifier(workspace).classify(["ls", "-l", "docs"], cwd=workspace)

    assert decision.allowed is False
    assert decision.code == "path_outside_workspace"


@pytest.mark.parametrize("path", [".env", ".env.local", "id_rsa", "config/token.txt"])
def test_denies_secret_like_paths(tmp_path: Path, path: str) -> None:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["head", "-n", "1", path], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"


@pytest.mark.parametrize("path", [".git/config", ".git/index"])
def test_denies_git_metadata_paths_for_direct_file_readers(tmp_path: Path, path: str) -> None:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("metadata\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["head", "-n", "1", path], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"


@pytest.mark.parametrize("argv", [["rm", "notes.md"], ["mv", "a", "b"], ["touch", "x"]])
def test_denies_write_commands(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "command_family_denied"


@pytest.mark.parametrize("argv", [["curl", "https://example.com"], ["wget", "https://example.com"]])
def test_denies_network_commands(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "command_family_denied"


@pytest.mark.parametrize("argv", [["python3", "-c", "1"], ["node", "-e", "1"], ["bash", "-c", "pwd"]])
def test_denies_interpreters(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "command_family_denied"


@pytest.mark.parametrize("argv", [["git", "add", "."], ["git", "commit", "-m", "x"], ["git", "push"]])
def test_denies_git_write_subcommands(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


@pytest.mark.parametrize(
    "argv",
    [["git", "branch", "new-branch"], ["git", "branch", "-c", "main", "copy"], ["git", "branch", "--track", "x"]],
)
def test_denies_git_branch_mutating_forms(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


@pytest.mark.parametrize("argv", [["git", "branch"], ["git", "branch", "--list"], ["git", "branch", "-a"]])
def test_allows_git_branch_read_only_forms(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "git.branch"


@pytest.mark.parametrize("argv", [["make", "test"], ["npm", "install"], ["docker", "ps"]])
def test_denies_package_build_runtime_managers(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "command_family_denied"


@pytest.mark.parametrize("argv", [["head"], ["tail"], ["wc", "-l"]])
def test_denies_stdin_reading_commands_without_path(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "path_argument_required"


def test_denies_tail_follow_mode(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("a\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["tail", "-f", "notes.md"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "unsupported_arguments"


def test_denies_rg_files_outside_workspace(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["rg", "--files", "/etc"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "path_outside_workspace"


def test_denies_rg_regexp_path_outside_workspace(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["rg", "-e", "needle", "/etc"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "path_outside_workspace"


def test_denies_git_branch_mutating_flags(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["git", "branch", "-D", "main"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


def test_denies_git_diff_no_index(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(
        ["git", "diff", "--no-index", "notes.md", "/etc/passwd"],
        cwd=tmp_path,
    )

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


def test_denies_secret_like_git_show_path(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["git", "show", "HEAD:.env"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"


@pytest.mark.parametrize(
    "flag",
    ["--pre", "-L", "--follow", "--hidden", "--no-ignore", "--passthru"],
)
def test_denies_rg_dangerous_or_unknown_flags(tmp_path: Path, flag: str) -> None:
    decision = _classifier(tmp_path).classify(["rg", flag, "needle", "notes.md"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "unsupported_arguments"


def test_shell_classification_redacts_option_values_in_audit_metadata(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(
        ["curl", "--header=Authorization: token-value"],
        cwd=tmp_path,
    )

    metadata = decision.to_tool_classification().metadata

    assert "Authorization" not in str(metadata)
    assert "token-value" not in str(metadata)
    assert metadata["argv"] == ["curl", "<option>"]


def test_shell_classification_redacts_unknown_command_token_in_audit_metadata(
    tmp_path: Path,
) -> None:
    decision = _classifier(tmp_path).classify(
        ["sk-secret-token", "--flag"],
        cwd=tmp_path,
    )

    metadata = decision.to_tool_classification().metadata

    assert "sk-secret-token" not in str(metadata)
    assert metadata["argv"] == ["<command>", "<option>"]


def test_denies_rg_full_workspace_secret_prone_scan(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["rg", ".", "."], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "path_argument_required"


def test_denies_rg_directory_with_secret_like_descendant(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "secrets.yml").write_text("secret\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["rg", "needle", "src"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"


def test_allows_rg_directory_without_secret_like_descendant(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("needle\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["rg", "needle", "docs"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == "rg"


@pytest.mark.parametrize("argv", [["git", "diff"], ["git", "log", "-p"], ["git", "show", "HEAD"]])
def test_denies_git_content_scans_without_explicit_safe_pathspec(
    tmp_path: Path,
    argv: list[str],
) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "status", "--short"],
        ["git", "ls-files", "--deleted"],
    ],
)
def test_denies_git_index_listing_without_explicit_safe_pathspec(
    tmp_path: Path,
    argv: list[str],
) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "status", "--short", "--", "docs"],
        ["git", "ls-files", "--deleted", "--", "docs"],
    ],
)
def test_denies_git_index_directory_pathspecs(
    tmp_path: Path,
    argv: list[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("hello\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "path_argument_must_be_file"


def test_shell_classification_redacts_secret_like_denied_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    secret_cwd = tmp_path / ".ssh"
    workspace.mkdir()
    secret_cwd.mkdir()

    decision = _classifier(workspace).classify(["pwd"], cwd=secret_cwd)
    metadata = decision.to_tool_classification().metadata

    assert decision.allowed is False
    assert str(secret_cwd) not in str(metadata)
    assert metadata["cwd"] == "<redacted>"


@pytest.mark.parametrize("argv", [["git", "diff", "--"], ["git", "show", "--"], ["git", "log", "-p", "--"]])
def test_denies_git_content_scans_with_empty_pathspec(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


def test_denies_git_directory_pathspec_with_secret_like_descendant(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "token.txt").write_text("secret\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["git", "diff", "--", "src"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"


def test_denies_git_content_directory_pathspec_even_without_current_secrets(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("hello\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["git", "diff", "--", "docs"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "path_argument_must_be_file"


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "show", "--stat", "--patch", "HEAD", "--", "docs/notes.md"],
        ["git", "show", "--stat", "-p", "HEAD", "--", "docs/notes.md"],
        ["git", "log", "-p", "--", "docs/notes.md"],
    ],
)
def test_denies_git_patch_producing_forms(tmp_path: Path, argv: list[str]) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("hello\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "git_subcommand_denied"


@pytest.mark.parametrize(
    "pathspec",
    [":/", ":(glob)src/**", "*.py"],
)
def test_denies_git_pathspec_magic(tmp_path: Path, pathspec: str) -> None:
    decision = _classifier(tmp_path).classify(
        ["git", "diff", "--", pathspec],
        cwd=tmp_path,
    )

    assert decision.allowed is False
    assert decision.code == "unsupported_arguments"


def test_subprocess_executor_kills_process_when_stdout_cap_is_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePipe:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = list(chunks)

        async def read(self, _limit: int) -> bytes:
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    class FakeProcess:
        def __init__(self) -> None:
                self.stdout = FakePipe([b"x" * 20])
                self.stderr = FakePipe([])
                self.returncode = None
                self.killed = False

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode

    process = FakeProcess()

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    monkeypatch.setattr(
        "assistant_core.tools.shell_read.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    executor = SubprocessShellExecutor(
        max_stdout_bytes=5,
        max_stderr_bytes=5,
        max_lines=10,
    )

    result = asyncio.run(
        executor.execute(
            argv=["pwd"],
            cwd=tmp_path,
            env={},
            timeout_seconds=1.0,
        ),
    )

    assert process.killed is True
    assert result.stdout == "xxxxx"
    assert result.stdout_truncated is True
    assert result.raw_stdout_bytes == 20


def test_subprocess_executor_kills_process_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingPipe:
        async def read(self, _limit: int) -> bytes:
            await asyncio.sleep(60)
            return b""

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = BlockingPipe()
            self.stderr = BlockingPipe()
            self.returncode = None
            self.killed = False

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode or 0

    process = FakeProcess()

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    monkeypatch.setattr(
        "assistant_core.tools.shell_read.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    executor = SubprocessShellExecutor(
        max_stdout_bytes=5,
        max_stderr_bytes=5,
        max_lines=10,
    )

    async def scenario() -> None:
        task = asyncio.create_task(
            executor.execute(
                argv=["pwd"],
                cwd=tmp_path,
                env={},
                timeout_seconds=10.0,
            ),
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert process.killed is True


def test_subprocess_executor_executes_resolved_absolute_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "rg"
    executable.write_text("", encoding="utf-8")

    class FakePipe:
        async def read(self, _limit: int) -> bytes:
            return b""

    class FakeProcess:
        stdout = FakePipe()
        stderr = FakePipe()
        returncode = 0

        def kill(self) -> None:
            raise AssertionError("should not kill")

        async def wait(self) -> int:
            return self.returncode

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("assistant_core.tools.shell_read.shutil.which", lambda command, path: str(bin_dir / command))
    monkeypatch.setattr("assistant_core.tools.shell_read._ALLOWED_EXECUTABLE_DIRS", frozenset({bin_dir}))
    monkeypatch.setattr("assistant_core.tools.shell_read._ALLOWED_EXECUTABLE_TARGET_ROOTS", frozenset({bin_dir}))
    monkeypatch.setattr(
        "assistant_core.tools.shell_read.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    executor = SubprocessShellExecutor()
    asyncio.run(
        executor.execute(
            argv=["rg", "needle", "docs/notes.md"],
            cwd=tmp_path,
            env={},
            timeout_seconds=1.0,
        ),
    )

    assert captured["args"][0] == str(executable)


def test_subprocess_executor_rejects_allowlisted_dir_symlink_to_untrusted_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    untrusted_dir = tmp_path / "untrusted"
    trusted_dir = tmp_path / "trusted"
    bin_dir.mkdir()
    untrusted_dir.mkdir()
    trusted_dir.mkdir()
    target = untrusted_dir / "rg"
    target.write_text("", encoding="utf-8")
    link = bin_dir / "rg"
    link.symlink_to(target)

    async def fail_create_subprocess_exec(*_args, **_kwargs):
        raise AssertionError("untrusted symlink target must not execute")

    monkeypatch.setattr("assistant_core.tools.shell_read.shutil.which", lambda command, path: str(bin_dir / command))
    monkeypatch.setattr("assistant_core.tools.shell_read._ALLOWED_EXECUTABLE_DIRS", frozenset({bin_dir}))
    monkeypatch.setattr("assistant_core.tools.shell_read._ALLOWED_EXECUTABLE_TARGET_ROOTS", frozenset({trusted_dir}))
    monkeypatch.setattr(
        "assistant_core.tools.shell_read.asyncio.create_subprocess_exec",
        fail_create_subprocess_exec,
    )

    executor = SubprocessShellExecutor()

    with pytest.raises(PermissionError):
        asyncio.run(
            executor.execute(
                argv=["rg", "needle", "docs/notes.md"],
                cwd=tmp_path,
                env={},
                timeout_seconds=1.0,
            ),
        )


def test_subprocess_executor_executes_resolved_symlink_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "bin"
    trusted_dir = tmp_path / "trusted"
    bin_dir.mkdir()
    trusted_dir.mkdir()
    target = trusted_dir / "rg"
    target.write_text("", encoding="utf-8")
    link = bin_dir / "rg"
    link.symlink_to(target)
    captured: dict[str, object] = {}

    class FakePipe:
        async def read(self, _limit: int) -> bytes:
            return b""

    class FakeProcess:
        stdout = FakePipe()
        stderr = FakePipe()
        returncode = 0

        def kill(self) -> None:
            raise AssertionError("should not kill")

        async def wait(self) -> int:
            return self.returncode

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("assistant_core.tools.shell_read.shutil.which", lambda command, path: str(bin_dir / command))
    monkeypatch.setattr("assistant_core.tools.shell_read._ALLOWED_EXECUTABLE_DIRS", frozenset({bin_dir}))
    monkeypatch.setattr("assistant_core.tools.shell_read._ALLOWED_EXECUTABLE_TARGET_ROOTS", frozenset({trusted_dir}))
    monkeypatch.setattr(
        "assistant_core.tools.shell_read.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    executor = SubprocessShellExecutor()
    asyncio.run(
        executor.execute(
            argv=["rg", "needle", "docs/notes.md"],
            cwd=tmp_path,
            env={},
            timeout_seconds=1.0,
        ),
    )

    assert captured["args"][0] == str(target)


def test_shell_tool_schema_caps_argv_shape(tmp_path: Path) -> None:
    from assistant_core.tools.shell_read import ProjectShellReadTool

    spec = ProjectShellReadTool(allowed_roots=[tmp_path]).spec
    argv_schema = spec.input_schema["properties"]["argv"]

    assert argv_schema["maxItems"] <= 16
    assert argv_schema["items"]["maxLength"] <= 256
