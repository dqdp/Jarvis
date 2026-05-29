from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Protocol, Sequence

from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolInvocationResult, ToolSpec
from assistant_core.tools.registry import ToolClassificationResult, ToolExecutionDenied


SHELL_READ_TOOL_NAME = "tool.shell.read.project"
_MINIMAL_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_EXTERNAL_DIFF": "",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PAGER": "cat",
    "PATH": "/Applications/Codex.app/Contents/Resources:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "LANG": "C",
    "LC_ALL": "C",
}
_SHELL_SYNTAX_MARKERS = ("|", ";", "&&", "||", ">", "<", "`", "$(", "\n", "\r")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SED_RANGE = re.compile(r"^(?P<start>[1-9][0-9]*)(,(?P<end>[1-9][0-9]*))?p$")

_DENIED_COMMANDS = {
    "rm",
    "mv",
    "cp",
    "chmod",
    "chown",
    "mkdir",
    "touch",
    "truncate",
    "tee",
    "dd",
    "sudo",
    "kill",
    "killall",
    "renice",
    "launchctl",
    "systemctl",
    "python",
    "python3",
    "node",
    "ruby",
    "perl",
    "bash",
    "sh",
    "zsh",
    "curl",
    "wget",
    "ssh",
    "scp",
    "nc",
    "telnet",
    "ftp",
    "docker",
    "make",
    "npm",
    "pnpm",
    "yarn",
    "pip",
    "pip3",
    "brew",
}
_ALLOWED_GIT_SUBCOMMANDS = {
    "status",
    "diff",
    "show",
    "log",
    "branch",
    "ls-files",
}
_DENIED_GIT_SUBCOMMANDS = {
    "add",
    "commit",
    "checkout",
    "reset",
    "clean",
    "push",
    "pull",
    "fetch",
    "merge",
    "rebase",
    "switch",
    "restore",
    "tag",
}
_ALLOWED_COMMANDS = {"pwd", "ls", "rg", "sed", "head", "tail", "wc", "git"}
_KNOWN_COMMANDS = _ALLOWED_COMMANDS | _DENIED_COMMANDS
_ALLOWED_EXECUTABLE_DIRS = frozenset(
    {
        Path("/bin"),
        Path("/Applications/Codex.app/Contents/Resources"),
        Path("/opt/homebrew/bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/usr/sbin"),
        Path("/sbin"),
    },
)
_ALLOWED_EXECUTABLE_TARGET_ROOTS = frozenset(
    {
        Path("/bin"),
        Path("/Applications/Codex.app/Contents/Resources"),
        Path("/opt/homebrew/Cellar"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/usr/sbin"),
        Path("/sbin"),
    },
)


@dataclass(frozen=True)
class ShellCommandDecision:
    allowed: bool
    code: str
    reason: str
    family: str | None = None
    argv: tuple[str, ...] = ()
    cwd: str | None = None
    metadata: dict[str, Any] | None = None

    def to_tool_classification(self) -> ToolClassificationResult:
        return ToolClassificationResult(
            allowed=self.allowed,
            code=self.code,
            reason=self.reason,
            metadata={
                "family": self.family,
                "cwd": _redacted_cwd(self.cwd, allowed=self.allowed),
                "argv": _redacted_argv(self.argv),
                **(self.metadata or {}),
            },
        )


@dataclass(frozen=True)
class ShellExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    raw_stdout_bytes: int | None = None
    raw_stderr_bytes: int | None = None


class ShellExecutionTimeout(TimeoutError):
    pass


class ShellExecutorPort(Protocol):
    async def execute(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> ShellExecutionResult: ...


class ShellCommandClassifier:
    def __init__(self, *, allowed_roots: Sequence[str | Path], max_lines: int = 200) -> None:
        self._allowed_roots = [_resolve_root(root) for root in allowed_roots]
        self._max_lines = max_lines

    def classify(self, argv: Sequence[str], *, cwd: str | Path) -> ShellCommandDecision:
        normalized = tuple(str(arg) for arg in argv)
        cwd_decision = self._resolve_cwd(cwd, normalized)
        if cwd_decision.allowed is False:
            return cwd_decision
        resolved_cwd = Path(cwd_decision.cwd or "")

        syntax_decision = _syntax_decision(normalized, resolved_cwd)
        if syntax_decision is not None:
            return syntax_decision

        if not normalized:
            return _deny("empty_command", "command argv must not be empty", normalized, resolved_cwd)

        if _command_is_path(normalized[0]):
            return _deny(
                "command_path_denied",
                "command must be an allowlisted bare executable name",
                normalized,
                resolved_cwd,
            )

        command = normalized[0]
        if command in _DENIED_COMMANDS:
            return _deny(
                "command_family_denied",
                "command family is denied",
                normalized,
                resolved_cwd,
            )

        if command == "pwd":
            return self._classify_pwd(normalized, resolved_cwd)
        if command == "ls":
            return self._classify_ls(normalized, resolved_cwd)
        if command == "rg":
            return self._classify_rg(normalized, resolved_cwd)
        if command == "sed":
            return self._classify_sed(normalized, resolved_cwd)
        if command in {"head", "tail"}:
            return self._classify_head_tail(command, normalized, resolved_cwd)
        if command == "wc":
            return self._classify_wc(normalized, resolved_cwd)
        if command == "git":
            return self._classify_git(normalized, resolved_cwd)

        return _deny("unsupported_command", "command is not allowlisted", normalized, resolved_cwd)

    def _resolve_cwd(
        self,
        cwd: str | Path,
        argv: tuple[str, ...],
    ) -> ShellCommandDecision:
        try:
            resolved = Path(cwd).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return _deny("invalid_working_directory", "working directory is invalid", argv, None)
        if not resolved.is_dir():
            return _deny(
                "invalid_working_directory",
                "working directory must be a directory",
                argv,
                resolved,
            )
        if _secret_like_path(str(resolved)):
            return _deny("secret_path_denied", "working directory is secret-like", argv, resolved)
        if not _is_inside_any_root(resolved, self._allowed_roots):
            return _deny(
                "path_outside_workspace",
                "working directory is outside allowed roots",
                argv,
                resolved,
            )
        return ShellCommandDecision(
            allowed=True,
            code="allowed",
            reason="working directory is allowed",
            argv=argv,
            cwd=str(resolved),
        )

    def _classify_pwd(self, argv: tuple[str, ...], cwd: Path) -> ShellCommandDecision:
        if len(argv) != 1:
            return _deny("unsupported_arguments", "pwd does not accept arguments", argv, cwd)
        return _allow("pwd", argv, cwd)

    def _classify_ls(self, argv: tuple[str, ...], cwd: Path) -> ShellCommandDecision:
        path_args = _ls_path_args(argv[1:])
        if path_args is None:
            return _deny("unsupported_arguments", "ls supports only non-recursive read-only flags", argv, cwd)
        if not path_args:
            path_args = ["."]
        path_decision = self._paths_decision(
            path_args,
            argv,
            cwd,
            recursive_directory_scan=False,
            include_hidden_listing=_ls_includes_hidden(argv[1:]),
            disclose_symlink_targets=_ls_discloses_symlink_targets(argv[1:]),
        )
        if path_decision is not None:
            return path_decision
        return _allow("ls", argv, cwd)

    def _classify_rg(self, argv: tuple[str, ...], cwd: Path) -> ShellCommandDecision:
        path_args = _rg_path_args(argv[1:])
        if path_args is None:
            return _deny("unsupported_arguments", "rg arguments are not allowlisted", argv, cwd)
        if not path_args or any(path in {".", "./"} for path in path_args):
            return _deny("path_argument_required", "explicit non-root rg path is required", argv, cwd)
        path_decision = self._paths_decision(
            path_args,
            argv,
            cwd,
        )
        if path_decision is not None:
            return path_decision
        return _allow("rg", argv, cwd)

    def _classify_sed(self, argv: tuple[str, ...], cwd: Path) -> ShellCommandDecision:
        if len(argv) < 4 or argv[1] != "-n":
            return _deny("unsupported_arguments", "sed is limited to sed -n RANGE FILE", argv, cwd)
        if not _bounded_sed_range(argv[2], self._max_lines):
            return _deny("line_range_exceeds_limit", "sed range exceeds line limit", argv, cwd)
        path_decision = self._paths_decision(list(argv[3:]), argv, cwd)
        if path_decision is not None:
            return path_decision
        return _allow("sed", argv, cwd)

    def _classify_head_tail(
        self,
        command: str,
        argv: tuple[str, ...],
        cwd: Path,
    ) -> ShellCommandDecision:
        line_count, path_args = _head_tail_args(argv[1:])
        if line_count is None:
            return _deny("unsupported_arguments", "head/tail supports only bounded -n reads", argv, cwd)
        if line_count > self._max_lines:
            return _deny("line_count_exceeds_limit", "line count exceeds limit", argv, cwd)
        if not path_args:
            return _deny("path_argument_required", "path argument is required", argv, cwd)
        path_decision = self._paths_decision(path_args, argv, cwd)
        if path_decision is not None:
            return path_decision
        return _allow(command, argv, cwd)

    def _classify_wc(self, argv: tuple[str, ...], cwd: Path) -> ShellCommandDecision:
        path_args = _wc_path_args(argv[1:])
        if path_args is None:
            return _deny("unsupported_arguments", "wc supports only read-only count flags", argv, cwd)
        if not path_args:
            return _deny("path_argument_required", "path argument is required", argv, cwd)
        path_decision = self._paths_decision(path_args, argv, cwd)
        if path_decision is not None:
            return path_decision
        return _allow("wc", argv, cwd)

    def _classify_git(self, argv: tuple[str, ...], cwd: Path) -> ShellCommandDecision:
        if len(argv) < 2:
            return _deny("unsupported_arguments", "git subcommand is required", argv, cwd)
        subcommand = argv[1]
        if subcommand in _DENIED_GIT_SUBCOMMANDS:
            return _deny("git_subcommand_denied", "git subcommand is denied", argv, cwd)
        if subcommand not in _ALLOWED_GIT_SUBCOMMANDS:
            return _deny("git_subcommand_denied", "git subcommand is not allowlisted", argv, cwd)
        if _secret_like_git_arg(argv[2:]):
            return _deny("secret_path_denied", "secret-like git paths are denied", argv, cwd)
        if not _git_args_read_only(subcommand, argv):
            return _deny("git_subcommand_denied", "git arguments are not allowlisted", argv, cwd)
        if _dangerous_git_args(subcommand, argv):
            return _deny("git_subcommand_denied", "git arguments are not read-only", argv, cwd)
        if subcommand == "branch" and not _git_branch_read_only(argv[2:]):
            return _deny("git_subcommand_denied", "git branch arguments are not read-only", argv, cwd)
        path_args = _git_path_args(subcommand, argv)
        if subcommand in {"diff", "show", "log"} and "--" in argv and not path_args:
            return _deny("git_subcommand_denied", "git pathspec is required", argv, cwd)
        if subcommand in {"status", "ls-files"} and not path_args:
            return _deny("git_subcommand_denied", "git pathspec is required", argv, cwd)
        path_decision = self._paths_decision(
            path_args,
            argv,
            cwd,
            require_files=subcommand in {"diff", "show", "log", "status", "ls-files"},
        )
        if path_decision is not None:
            return path_decision
        return _allow(f"git.{subcommand}", argv, cwd)

    def _paths_decision(
        self,
        path_args: list[str],
        argv: tuple[str, ...],
        cwd: Path,
        *,
        require_files: bool = False,
        recursive_directory_scan: bool = True,
        include_hidden_listing: bool = True,
        disclose_symlink_targets: bool = True,
    ) -> ShellCommandDecision | None:
        for raw_path in path_args:
            if raw_path.startswith("-"):
                return _deny("unsupported_arguments", "path argument cannot look like an option", argv, cwd)
            if _has_pathspec_magic(raw_path):
                return _deny("unsupported_arguments", "pathspec magic is denied", argv, cwd)
            if _secret_like_path(raw_path):
                return _deny("secret_path_denied", "secret-like paths are denied", argv, cwd)
            try:
                resolved = _resolve_command_path(raw_path, cwd)
            except (OSError, RuntimeError, ValueError):
                return _deny("invalid_path", "path argument is invalid", argv, cwd)
            if not _is_inside_any_root(resolved, self._allowed_roots):
                return _deny("path_outside_workspace", "path escapes allowed roots", argv, cwd)
            if _secret_like_path(str(resolved)):
                return _deny("secret_path_denied", "secret-like paths are denied", argv, cwd)
            if resolved.is_dir():
                if recursive_directory_scan:
                    descendant_decision = _descendant_paths_decision(
                        resolved,
                        allowed_roots=self._allowed_roots,
                        argv=argv,
                        cwd=cwd,
                    )
                    if descendant_decision is not None:
                        return descendant_decision
                else:
                    listing_decision = _directory_listing_decision(
                        resolved,
                        allowed_roots=self._allowed_roots,
                        argv=argv,
                        cwd=cwd,
                        include_hidden=include_hidden_listing,
                        disclose_symlink_targets=disclose_symlink_targets,
                    )
                    if listing_decision is not None:
                        return listing_decision
            if require_files and not resolved.is_file():
                return _deny("path_argument_must_be_file", "path argument must be a file", argv, cwd)
        return None


class SubprocessShellExecutor:
    def __init__(
        self,
        *,
        max_stdout_bytes: int = 20_000,
        max_stderr_bytes: int = 20_000,
        max_lines: int = 200,
    ) -> None:
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._max_lines = max_lines

    async def execute(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> ShellExecutionResult:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *_resolve_executable_argv(argv, env),
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(
                    _read_limited_pipe(
                        process.stdout,
                        process=process,
                        max_bytes=self._max_stdout_bytes,
                        max_lines=self._max_lines,
                    ),
                    _read_limited_pipe(
                        process.stderr,
                        process=process,
                        max_bytes=self._max_stderr_bytes,
                        max_lines=self._max_lines,
                    ),
                ),
                timeout=timeout_seconds,
            )
            await process.wait()
        except TimeoutError as exc:
            if process is not None:
                await _kill_process(process)
            raise ShellExecutionTimeout("shell command timed out") from exc
        except asyncio.CancelledError:
            if process is not None:
                await _kill_process(process)
            raise
        return ShellExecutionResult(
            exit_code=int(process.returncode or 0),
            stdout=stdout.text,
            stderr=stderr.text,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            raw_stdout_bytes=stdout.raw_bytes,
            raw_stderr_bytes=stderr.raw_bytes,
        )


class ProjectShellReadTool:
    content_type = "application/json"

    def __init__(
        self,
        *,
        allowed_roots: Sequence[str | Path],
        executor: ShellExecutorPort | None = None,
        max_stdout_bytes: int = 20_000,
        max_stderr_bytes: int = 20_000,
        max_lines: int = 200,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._allowed_roots = [_resolve_root(root) for root in allowed_roots]
        self._classifier = ShellCommandClassifier(
            allowed_roots=self._allowed_roots,
            max_lines=max_lines,
        )
        self._executor = executor or SubprocessShellExecutor(
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            max_lines=max_lines,
        )
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._max_lines = max_lines
        self._timeout_seconds = timeout_seconds
        self.spec = ToolSpec(
            name=SHELL_READ_TOOL_NAME,
            display_name="Project Shell Read",
            description="Runs allowlisted read-only project inspection commands.",
            capability=Capability.TOOL_SHELL_READ,
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 256},
                        "minItems": 1,
                        "maxItems": 16,
                        "maxLength": 256,
                    },
                    "cwd": {"type": "string", "maxLength": 512},
                },
                "required": ["argv", "cwd"],
                "additionalProperties": False,
            },
            adapter_name="shell.project_read",
            output_schema={
                "type": "object",
                "properties": {
                    "exit_code": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                },
            },
            default_timeout_seconds=timeout_seconds,
            max_output_bytes=max_stdout_bytes + max_stderr_bytes + 2048,
            sensitivity_ceiling=Sensitivity.PROJECT,
        )

    def classify(self, arguments: dict[str, Any]) -> ToolClassificationResult:
        argv = _arguments_argv(arguments)
        cwd = _arguments_cwd(arguments)
        return self._classifier.classify(argv, cwd=cwd).to_tool_classification()

    async def invoke(self, arguments: dict[str, Any]) -> ToolInvocationResult:
        argv = _arguments_argv(arguments)
        cwd = Path(_arguments_cwd(arguments)).expanduser().resolve(strict=True)
        decision = self._classifier.classify(argv, cwd=cwd)
        if not decision.allowed:
            raise ToolExecutionDenied(
                decision.code,
                decision.reason,
                metadata=decision.to_tool_classification().metadata,
            )
        result = await self._executor.execute(
            argv=_execution_argv(decision.argv),
            cwd=cwd,
            env=dict(_MINIMAL_ENV),
            timeout_seconds=self._timeout_seconds,
        )
        stdout, stdout_metadata = _bounded_text(
            result.stdout,
            max_bytes=self._max_stdout_bytes,
            max_lines=self._max_lines,
        )
        stderr, stderr_metadata = _bounded_text(
            result.stderr,
            max_bytes=self._max_stderr_bytes,
            max_lines=self._max_lines,
        )
        stdout_truncated = bool(stdout_metadata["truncated"] or result.stdout_truncated)
        stderr_truncated = bool(stderr_metadata["truncated"] or result.stderr_truncated)
        truncated = stdout_truncated or stderr_truncated
        content = {
            "exit_code": result.exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": {
                "stdout": stdout_truncated,
                "stderr": stderr_truncated,
            },
        }
        raw_stdout_bytes = (
            result.raw_stdout_bytes
            if result.raw_stdout_bytes is not None
            else int(stdout_metadata["raw_bytes"])
        )
        raw_stderr_bytes = (
            result.raw_stderr_bytes
            if result.raw_stderr_bytes is not None
            else int(stderr_metadata["raw_bytes"])
        )
        raw_output_bytes = raw_stdout_bytes + raw_stderr_bytes
        if _secret_like_output(result.stdout) or _secret_like_output(result.stderr):
            return ToolInvocationResult(
                content=json.dumps({"redacted": True}, sort_keys=True),
                content_type="application/json",
                truncated=truncated,
                output_bytes=raw_output_bytes,
                metadata={
                    "family": decision.family,
                    "cwd": decision.cwd,
                    "exit_code": result.exit_code,
                    "redacted": True,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "raw_stdout_bytes": raw_stdout_bytes,
                    "raw_stderr_bytes": raw_stderr_bytes,
                },
            )
        return ToolInvocationResult(
            content=json.dumps(content, sort_keys=True),
            content_type="application/json",
            truncated=truncated,
            output_bytes=raw_output_bytes,
            metadata={
                "family": decision.family,
                "cwd": decision.cwd,
                "exit_code": result.exit_code,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "raw_stdout_bytes": raw_stdout_bytes,
                "raw_stderr_bytes": raw_stderr_bytes,
            },
        )


def project_shell_read_tool_from_config(capabilities: dict[str, Any]) -> ProjectShellReadTool:
    config = capabilities["tool.shell.read"]
    max_output_bytes = int(config["max_output_bytes"])
    return ProjectShellReadTool(
        allowed_roots=[Path(root) for root in config["allowed_roots"]],
        max_stdout_bytes=max_output_bytes // 2,
        max_stderr_bytes=max_output_bytes - (max_output_bytes // 2),
        max_lines=int(config.get("max_lines", 200)),
        timeout_seconds=float(config["timeout_seconds"]),
    )


def _arguments_argv(arguments: dict[str, Any]) -> list[str]:
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("argv must be a list of strings")
    return argv


def _arguments_cwd(arguments: dict[str, Any]) -> str:
    cwd = arguments.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("cwd must be a non-empty string")
    return cwd


def _resolve_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve(strict=True)


def _syntax_decision(
    argv: tuple[str, ...],
    cwd: Path,
) -> ShellCommandDecision | None:
    if not argv:
        return None
    if _ENV_ASSIGNMENT.match(argv[0]):
        return _deny("shell_syntax_denied", "environment assignment prefixes are denied", argv, cwd)
    if argv[0] == "sh" and "-c" in argv:
        return _deny("shell_syntax_denied", "shell execution is denied", argv, cwd)
    for arg in argv:
        if any(marker in arg for marker in _SHELL_SYNTAX_MARKERS):
            return _deny("shell_syntax_denied", "shell metacharacters are denied", argv, cwd)
    return None


def _command_is_path(command: str) -> bool:
    return (
        "/" in command
        or "\\" in command
        or command in {".", ".."}
        or command.startswith(".")
        or Path(command).is_absolute()
    )


def _has_pathspec_magic(raw_path: str) -> bool:
    return raw_path.startswith(":") or any(character in raw_path for character in "*?[")


@dataclass(frozen=True)
class _LimitedOutput:
    text: str
    raw_bytes: int
    truncated: bool


def _rg_path_args(args: Sequence[str]) -> list[str] | None:
    path_args: list[str] = []
    saw_pattern = any(arg == "--files" for arg in args)
    skip_next = False
    pattern_options = {"-e", "--regexp"}
    allowed_flags = {
        "--files",
        "--fixed-strings",
        "--ignore-case",
        "--line-number",
        "--no-heading",
        "--smart-case",
        "--with-filename",
        "-F",
        "-H",
        "-S",
        "-i",
        "-n",
    }
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in pattern_options:
            saw_pattern = True
            skip_next = True
            continue
        if arg.startswith("--regexp="):
            saw_pattern = True
            continue
        if arg in allowed_flags:
            continue
        if arg.startswith("-"):
            return None
        if not saw_pattern:
            saw_pattern = True
            continue
        path_args.append(arg)
    return path_args


def _bounded_sed_range(raw_range: str, max_lines: int) -> bool:
    match = _SED_RANGE.match(raw_range)
    if match is None:
        return False
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    return end >= start and (end - start + 1) <= max_lines


def _head_tail_args(args: Sequence[str]) -> tuple[int | None, list[str]]:
    if not args:
        return 10, []
    if args[0] == "-n":
        if len(args) < 2:
            return None, []
        try:
            count = int(args[1])
        except ValueError:
            return None, []
        return count if count > 0 else None, list(args[2:])
    if args[0].startswith("-") and args[0][1:].isdigit():
        return int(args[0][1:]), list(args[1:])
    if args[0].startswith("-"):
        return None, []
    return 10, list(args)


def _ls_path_args(args: Sequence[str]) -> list[str] | None:
    allowed_short = set("1ACFSTacdfghiklmnopqrstux")
    allowed_long = {
        "--almost-all",
        "--classify",
        "--color=never",
        "--directory",
        "--file-type",
        "--format=long",
        "--human-readable",
        "--indicator-style=slash",
        "--numeric-uid-gid",
        "--reverse",
        "--sort=name",
    }
    path_args: list[str] = []
    for arg in args:
        if arg.startswith("--"):
            if arg in allowed_long:
                continue
            return None
        if arg.startswith("-") and arg != "-":
            if any(character in {"L", "R"} for character in arg[1:]):
                return None
            if all(character in allowed_short for character in arg[1:]):
                continue
            return None
        path_args.append(arg)
    return path_args


def _ls_includes_hidden(args: Sequence[str]) -> bool:
    for arg in args:
        if arg in {"--almost-all"}:
            return True
        if arg.startswith("-") and not arg.startswith("--") and any(character in {"A", "a"} for character in arg[1:]):
            return True
    return False


def _ls_discloses_symlink_targets(args: Sequence[str]) -> bool:
    for arg in args:
        if arg == "--format=long":
            return True
        if arg.startswith("-") and not arg.startswith("--") and "l" in arg[1:]:
            return True
    return False


def _wc_path_args(args: Sequence[str]) -> list[str] | None:
    allowed_flags = {"-c", "-l", "-m", "-w"}
    path_args: list[str] = []
    for arg in args:
        if arg.startswith("-"):
            if arg in allowed_flags:
                continue
            if len(arg) > 2 and all(f"-{character}" in allowed_flags for character in arg[1:]):
                continue
            return None
        path_args.append(arg)
    return path_args


def _dangerous_git_args(subcommand: str, argv: tuple[str, ...]) -> bool:
    args = argv[2:]
    if subcommand == "diff" and "--" not in argv:
        return True
    if subcommand == "show" and "--stat" not in args and "--" not in argv:
        return True
    if subcommand == "log" and any(arg in {"-p", "--patch"} for arg in args) and "--" not in argv:
        return True
    if any(arg == "--no-index" for arg in args) and subcommand == "diff":
        return True
    if any(arg == "--output" or arg.startswith("--output=") for arg in args):
        return True
    if subcommand == "branch":
        dangerous_branch_flags = {
            "-d",
            "-D",
            "-m",
            "-M",
            "-u",
            "--delete",
            "--move",
            "--set-upstream-to",
            "--unset-upstream",
        }
        return any(
            arg in dangerous_branch_flags
            or any(arg.startswith(f"{flag}=") for flag in dangerous_branch_flags)
            for arg in args
        )
    return False


def _git_args_read_only(subcommand: str, argv: tuple[str, ...]) -> bool:
    args = argv[2:]
    if subcommand == "status":
        return _args_match_read_only_spec(
            args,
            allowed_flags={"--branch", "--porcelain", "--short", "-b", "-s"},
            allow_revisions=False,
            allow_pathspec=True,
        )
    if subcommand == "diff":
        return _args_match_read_only_spec(
            args,
            allowed_flags={
                "--cached",
                "--color=never",
                "--name-only",
                "--name-status",
                "--no-ext-diff",
                "--no-textconv",
                "--stat",
            },
            allow_revisions=False,
            allow_pathspec=True,
        )
    if subcommand == "show":
        if "--patch" in args or "-p" in args:
            return False
        if "--stat" not in args or "--no-patch" not in args:
            return False
        return _args_match_read_only_spec(
            args,
            allowed_flags={"--color=never", "--no-patch", "--no-textconv", "--stat"},
            allow_revisions=True,
            allow_pathspec=True,
        )
    if subcommand == "log":
        if "--patch" in args or "-p" in args or "--stat" in args:
            return False
        return _args_match_read_only_spec(
            args,
            allowed_flags={"--color=never", "--decorate=short", "--no-patch", "--oneline"},
            allowed_value_flags={"--max-count", "-n"},
            allow_revisions=True,
            allow_pathspec=True,
        )
    if subcommand == "branch":
        return True
    if subcommand == "ls-files":
        return _args_match_read_only_spec(
            args,
            allowed_flags={"--cached", "--deleted", "--modified", "--others", "--stage"},
            allow_revisions=False,
            allow_pathspec=True,
        )
    return False


def _args_match_read_only_spec(
    args: Sequence[str],
    *,
    allowed_flags: set[str],
    allowed_value_flags: set[str] | None = None,
    allow_revisions: bool,
    allow_pathspec: bool,
) -> bool:
    allowed_value_flags = allowed_value_flags or set()
    after_path_separator = False
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            if not allow_pathspec:
                return False
            after_path_separator = True
            index += 1
            continue
        if after_path_separator:
            if arg.startswith("-"):
                return False
            index += 1
            continue
        if arg in allowed_flags:
            index += 1
            continue
        if arg in allowed_value_flags:
            if index + 1 >= len(args):
                return False
            index += 2
            continue
        if any(arg.startswith(f"{flag}=") for flag in allowed_value_flags):
            index += 1
            continue
        if arg.startswith("-"):
            return False
        if allow_revisions and _looks_like_git_revision(arg):
            index += 1
            continue
        return False
    return True


def _looks_like_git_revision(value: str) -> bool:
    if _secret_like_path(value) or _has_pathspec_magic(value):
        return False
    if ":" in value or "/" in value or "\\" in value:
        return False
    return bool(value) and all(character.isalnum() or character in "._-~^" for character in value)


def _git_branch_read_only(args: Sequence[str]) -> bool:
    if not args:
        return True
    allowed_no_value = {"--all", "--list", "--remotes", "--show-current", "-a", "-r"}
    allowed_with_value = {"--contains", "--merged", "--no-merged", "--points-at"}
    index = 0
    saw_list_mode = False
    while index < len(args):
        arg = args[index]
        if arg in allowed_no_value:
            saw_list_mode = True
            index += 1
            continue
        if arg in allowed_with_value:
            if index + 1 >= len(args):
                return False
            saw_list_mode = True
            index += 2
            continue
        if any(arg.startswith(f"{flag}=") for flag in allowed_with_value):
            saw_list_mode = True
            index += 1
            continue
        if arg.startswith("-"):
            return False
        if saw_list_mode:
            index += 1
            continue
        return False
    return True


def _secret_like_git_arg(args: Sequence[str]) -> bool:
    for arg in args:
        if _secret_like_path(arg):
            return True
        if ":" in arg:
            _ref, tree_path = arg.split(":", 1)
            if tree_path and _secret_like_path(tree_path):
                return True
    return False


def _git_path_args(subcommand: str, argv: tuple[str, ...]) -> list[str]:
    if "--" in argv:
        return list(argv[argv.index("--") + 1 :])
    if subcommand not in {"diff", "log", "status", "ls-files"}:
        return []
    path_args: list[str] = []
    for arg in argv[2:]:
        if arg.startswith("-"):
            continue
        path = Path(arg)
        if path.is_absolute() or arg.startswith(".") or "/" in arg:
            path_args.append(arg)
    return path_args


def _resolve_command_path(raw_path: str, cwd: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = cwd / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        return path.resolve(strict=False)


def _secret_like_path(raw_path: str) -> bool:
    lowered = raw_path.lower()
    parts = [part.lower() for part in Path(raw_path).parts]
    if any(part == ".env" or part.startswith(".env.") for part in parts):
        return True
    if any(part in {".git", ".ssh", "id_rsa", "id_ed25519", "known_hosts"} for part in parts):
        return True
    if any(part in {".aws", ".gcp", ".azure", ".kube"} for part in parts):
        return True
    if lowered.endswith((".pem", ".key", ".crt")):
        return True
    return any(marker in lowered for marker in ("credentials", "token", "secrets"))


def _secret_like_output(output: str) -> bool:
    return any(_secret_like_path(token.strip('"\':,;()[]{}')) for token in output.split())


def _execution_argv(argv: tuple[str, ...]) -> list[str]:
    if not argv or argv[0] != "git":
        return list(argv)
    subcommand = argv[1] if len(argv) > 1 else ""
    hardened: list[str] = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.pager=cat",
        "-c",
        "diff.external=",
        "-c",
        "pager.show=false",
        "--no-pager",
    ]
    if subcommand in {"diff", "log", "show"}:
        hardened.extend([subcommand, "--no-ext-diff", "--no-textconv", *argv[2:]])
    else:
        hardened.extend(argv[1:])
    return hardened


def _resolve_executable_argv(argv: list[str], env: dict[str, str]) -> list[str]:
    if not argv:
        return argv
    executable = shutil.which(argv[0], path=env.get("PATH") or _MINIMAL_ENV["PATH"])
    if executable is None:
        raise FileNotFoundError(f"allowed executable not found: {argv[0]}")
    found = Path(executable).expanduser()
    if not found.is_absolute():
        raise PermissionError("resolved executable must be absolute")
    if found.name != argv[0] or found.parent not in _ALLOWED_EXECUTABLE_DIRS:
        raise PermissionError("resolved executable is outside allowed directories")
    target = found.resolve(strict=True)
    if target.name != argv[0]:
        raise PermissionError("resolved executable target name does not match command")
    if not _is_inside_any_root(target, _ALLOWED_EXECUTABLE_TARGET_ROOTS):
        raise PermissionError("resolved executable target is outside allowed roots")
    return [str(target), *argv[1:]]


def _descendant_paths_decision(
    path: Path,
    *,
    allowed_roots: Sequence[Path],
    argv: tuple[str, ...],
    cwd: Path,
    max_entries: int = 5000,
) -> ShellCommandDecision | None:
    try:
        for index, descendant in enumerate(path.rglob("*")):
            if index >= max_entries:
                return _deny("secret_path_denied", "too many descendant paths to classify safely", argv, cwd)
            if _secret_like_path(str(descendant.relative_to(path))):
                return _deny("secret_path_denied", "secret-like descendant paths are denied", argv, cwd)
            if descendant.is_symlink():
                target = descendant.resolve(strict=True)
                if not _is_inside_any_root(target, allowed_roots):
                    return _deny("path_outside_workspace", "symlink descendant escapes allowed roots", argv, cwd)
                if _secret_like_path(str(target)):
                    return _deny("secret_path_denied", "secret-like symlink target is denied", argv, cwd)
    except (OSError, RuntimeError, ValueError):
        return _deny("secret_path_denied", "descendant paths could not be classified safely", argv, cwd)
    return None


def _directory_listing_decision(
    path: Path,
    *,
    allowed_roots: Sequence[Path],
    argv: tuple[str, ...],
    cwd: Path,
    include_hidden: bool,
    disclose_symlink_targets: bool,
) -> ShellCommandDecision | None:
    try:
        for child in path.iterdir():
            if child.name.startswith(".") and not include_hidden:
                continue
            if _secret_like_path(child.name):
                return _deny("secret_path_denied", "secret-like directory entries are denied", argv, cwd)
            if child.is_symlink() and disclose_symlink_targets:
                target = child.resolve(strict=True)
                if not _is_inside_any_root(target, allowed_roots):
                    return _deny("path_outside_workspace", "symlink entry escapes allowed roots", argv, cwd)
                if _secret_like_path(str(target)):
                    return _deny("secret_path_denied", "secret-like symlink target is denied", argv, cwd)
    except (OSError, RuntimeError, ValueError):
        return _deny("secret_path_denied", "directory entries could not be classified safely", argv, cwd)
    return None


def _bounded_text(text: str, *, max_bytes: int, max_lines: int) -> tuple[str, dict[str, int | bool]]:
    raw_bytes = len(text.encode("utf-8"))
    truncated = False
    lines = text.splitlines(keepends=True)
    if len(lines) > max_lines:
        text = "".join(lines[:max_lines])
        truncated = True
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        text = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True
    return text, {"raw_bytes": raw_bytes, "truncated": truncated}


async def _read_limited_pipe(
    pipe: asyncio.StreamReader,
    *,
    process: asyncio.subprocess.Process,
    max_bytes: int,
    max_lines: int,
) -> _LimitedOutput:
    raw = bytearray()
    raw_bytes = 0
    truncated = False
    while True:
        chunk = await pipe.read(min(4096, max_bytes + 1))
        if not chunk:
            break
        raw.extend(chunk)
        raw_bytes += len(chunk)
        if raw_bytes > max_bytes or raw.count(b"\n") > max_lines:
            truncated = True
            _kill_process_now(process)
            break
    bounded = _truncate_output_bytes(bytes(raw), max_bytes=max_bytes, max_lines=max_lines)
    if len(bounded) < len(raw):
        truncated = True
    return _LimitedOutput(
        text=bounded.decode("utf-8", errors="replace"),
        raw_bytes=raw_bytes,
        truncated=truncated,
    )


def _truncate_output_bytes(raw: bytes, *, max_bytes: int, max_lines: int) -> bytes:
    lines = raw.splitlines(keepends=True)
    if len(lines) > max_lines:
        raw = b"".join(lines[:max_lines])
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw


def _kill_process_now(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    _kill_process_now(process)
    await process.wait()


def _allow(family: str, argv: tuple[str, ...], cwd: Path) -> ShellCommandDecision:
    return ShellCommandDecision(
        allowed=True,
        code="allowed",
        reason="command is allowed",
        family=family,
        argv=argv,
        cwd=str(cwd),
    )


def _deny(
    code: str,
    reason: str,
    argv: tuple[str, ...],
    cwd: Path | None,
) -> ShellCommandDecision:
    return ShellCommandDecision(
        allowed=False,
        code=code,
        reason=reason,
        argv=argv,
        cwd=str(cwd) if cwd is not None else None,
    )


def _is_inside_any_root(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _redacted_argv(argv: tuple[str, ...]) -> list[str]:
    if not argv:
        return []
    redacted = [argv[0] if argv[0] in _KNOWN_COMMANDS and not _command_is_path(argv[0]) else "<command>"]
    for arg in argv[1:]:
        redacted.append("<option>" if arg.startswith("-") else "<arg>")
    return redacted


def _redacted_cwd(cwd: str | None, *, allowed: bool) -> str | None:
    if cwd is None:
        return None
    if not allowed or _secret_like_path(cwd):
        return "<redacted>"
    return cwd
