# ADR-033 — Shell Sandbox and Local Command Policy

## Status

Accepted.

## Context

PM-06 introduces local command execution as a post-MVP tool capability.

This is useful for:

- project inspection;
- code and documentation search;
- git status/diff review;
- local runtime diagnostics;
- model/server/resource troubleshooting.

It is also one of the highest-risk early capabilities. A shell tool can read
private files, reveal secrets, access network state, invoke interpreters, write
files or mutate git state if it is implemented as arbitrary terminal access.

ADR-029 defines the capability and permission model. ADR-030 defines
`ToolGatewayPort` as the only tool execution boundary. ADR-031 defines loop
strategies. This ADR defines the sandbox and command policy required before any
shell-like execution is added.

## Decision

PM-06 shell execution is split into two read-only slices:

```text
PM-06a Project read-only shell tool
PM-06b Read-only system diagnostics tools
```

Both slices execute through `ToolGatewayPort`.

No loop strategy, CLI handler, API handler or runtime module may execute local
commands directly.

The baseline policy is:

```text
argv only
no shell
allowlist command families
allowlist working roots
deny writes
deny destructive commands
deny network clients
deny interpreters
deny secret-like paths
bound output
bound wall-clock time
audit every allow, deny, timeout and truncation
```

## Capability model

Use separate capabilities for project inspection and system diagnostics:

```text
tool.shell.read.project
tool.system.read.process
tool.system.read.resources
tool.system.read.hardware
tool.system.read.network
tool.system.read.sensors
```

`tool.shell.write`, unrestricted shell, network clients and destructive local
actions are not part of PM-06.

## Command execution model

Commands must be represented as an argv array:

```text
["rg", "ToolGateway", "docs"]
["git", "status", "--short"]
["sed", "-n", "1,120p", "docs/file.md"]
```

Commands must not be represented as a shell string:

```text
sh -c "rg ToolGateway docs | head"
```

Disallow:

```text
sh -c
bash -c
zsh -c
shell metacharacters
pipes
redirection
command separators
subshells
environment assignment prefixes
```

This makes classification deterministic before execution.

## PM-06a project inspection allowlist

Initial project read-only commands:

```text
pwd
ls
rg
sed -n
head
tail
wc
git status
git diff
git show
git log
git branch
git ls-files
```

`cat` is intentionally excluded from the first slice. File reads should use
bounded commands such as `sed -n`, `head` or `tail`.

Allowed git subcommands must remain read-only. Write or state-changing git
subcommands are denied even if the command starts with `git`.

## PM-06b system diagnostics allowlist

Initial cross-platform read-only diagnostics:

```text
ps
pgrep
uptime
df
du
```

`du` is allowed only for paths inside an allowlisted workspace root.

macOS read-only diagnostics:

```text
top -l 1
vm_stat
sysctl selected read-only keys
netstat selected read-only flags
ifconfig
lsof selected read-only flags
```

Linux read-only diagnostics:

```text
top -b -n 1
free
lscpu
lshw
ss selected read-only flags
netstat selected read-only flags
ip addr
lsof selected read-only flags
nvidia-smi
```

Temperature and sensor diagnostics:

```text
macOS: powermetrics --samplers smc -n 1 when available without sudo
Linux: sensors
Linux: read-only /sys/class/thermal/thermal_zone*/temp adapter
Linux GPU: nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

Sensor diagnostics are one-shot snapshots. They must not poll indefinitely,
change fan settings, change power settings, write to `/sys` or request
privilege escalation.

If a sensor backend is absent or requires `sudo`, the tool returns an
`unavailable` or `denied` observation instead of prompting for a password.

Diagnostics tools must run in non-interactive mode. Interactive tools such as
`htop`, `less`, `vim` and `watch` are denied by default unless a later ADR
defines a non-interactive adapter for them.

Network diagnostics are read-only but privacy-sensitive. Their output must be
bounded and redacted where appropriate.

## Denied command families

Denied in PM-06:

```text
rm
mv
cp
chmod
chown
mkdir
touch
truncate
tee
dd
sudo
kill
killall
renice
launchctl
systemctl
python
python3
node
ruby
perl
bash
sh
zsh
curl
wget
ssh
scp
nc
telnet
ftp
git add
git commit
git checkout
git reset
git clean
git push
git pull
git fetch
docker
make
npm
pnpm
yarn
pip
pip3
brew
```

Some denied families may become later capabilities with stricter approval,
sandboxing and tests. They are not part of PM-06.

## Working-directory and path policy

Every command has an allowlisted working directory.

Rules:

```text
cwd must be inside an allowlisted root
relative paths are resolved against cwd
absolute paths must stay inside an allowlisted root unless explicitly allowed
path traversal outside allowlisted roots is denied
symlink traversal outside allowlisted roots is denied
```

PM-06a defaults to project roots only.

PM-06b may read system state through approved diagnostics commands, but file
path arguments remain restricted unless the command family explicitly permits a
system read.

## Secret path policy

Secret-like paths are denied by default:

```text
.env
.env.*
*.pem
*.key
*.crt
id_rsa
id_ed25519
known_hosts
credentials
token
secrets
.aws
.gcp
.azure
.kube
```

The first shell slices deny secret-like paths instead of trying to classify and
partially redact their contents.

Process command lines and network diagnostics may also contain credentials.
Those outputs must be truncated and redacted before they are returned to the
model or persisted in events.

## Output, timeout and environment rules

Each invocation must define:

```text
max_stdout_bytes
max_stderr_bytes
max_lines
max_wall_time_seconds
```

If output exceeds limits, return a structured truncated observation with
truncation metadata.

The tool must not return raw environment variables. Environment passed to the
subprocess should be minimal and explicit.

## Permission modes

Default PM-06 behavior by permission mode:

```text
developer_local:
  project read -> allow
  system process/resource/hardware read -> allow
  system network diagnostics -> allow with redaction and output caps

locked_down:
  project read -> approval_required
  system diagnostics -> approval_required

automation:
  only explicitly configured command families and roots
```

Write shell is not available in PM-06 under any permission mode.

## Events and audit

Every shell or diagnostics attempt emits auditable events:

```text
tool.shell.classified
tool.shell.denied
tool.shell.started
tool.shell.completed
tool.shell.failed
tool.shell.timeout
tool.shell.output_truncated
```

Event payloads must include:

```text
capability
command family
argv redacted
cwd
policy outcome
exit code when available
duration
truncation metadata
request_id/correlation_id when available
```

Events must not include raw secrets, full unbounded stdout/stderr or raw
environment values.

## Testing requirements

PM-06a must include unit tests for:

```text
allowed project commands
denied shell metacharacters
denied path traversal
denied secret-like paths
denied write commands
denied network commands
denied interpreters
denied git write subcommands
```

PM-06a must include contract tests for:

```text
allowed argv execution
bounded stdout/stderr
truncated output metadata
timeout behavior
audit events
denied observation without execution
```

PM-06b must include unit and contract tests for:

```text
allowed process/resource/hardware diagnostics
allowed network diagnostics with redaction
allowed temperature sensor snapshot
temperature readings normalized to Celsius when possible
missing sensor backend returns unavailable without failing the whole flow
sudo or privilege-required sensor command is denied
sensor write paths are denied
long-running sensor polling is denied
denied interactive diagnostics
du restricted to workspace roots
platform-specific command classification
bounded output
timeouts
audit events
```

Architecture tests must ensure:

```text
AgentRuntime does not import subprocess
LoopStrategy implementations do not import subprocess
only shell adapters execute subprocess
ToolGateway consults PolicyPort before shell execution
```

CI must use fake shell/diagnostics adapters where host state would make tests
non-deterministic.

## Consequences

Benefits:

- useful local project inspection without arbitrary terminal access;
- clear security boundary before shell tools are exposed to models;
- PM-06 can be implemented test-first and incrementally;
- diagnostics can support local model/runtime troubleshooting;
- future write/network shell capabilities have a defined place to extend.

Costs:

- command coverage is intentionally small at first;
- some convenient shell idioms such as pipes and redirects are not available;
- diagnostics need platform-specific classification;
- output redaction and truncation add adapter complexity.

## Non-goals

ADR-033 does not introduce:

```text
write shell
destructive shell
arbitrary bash/zsh
interactive terminal sessions
network clients
package managers
build tools
remote execution
MCP gateway
container sandbox
secret manager access
```

Those require later ADRs or slice-plan updates.
