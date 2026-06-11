#!/usr/bin/env python3
# agent0.py - AI Agent with memory, tool feedback, and security controls
# Run: python agent0.py

import subprocess
import os
import re
import json
import shlex
import glob as glob_mod
import urllib.request
from pathlib import Path, PureWindowsPath
from dataclasses import dataclass, field
from enum import IntEnum

# ─── Configuration ───

WORKSPACE = os.path.expanduser("~/.agent0")
CONFIG_PATH = os.path.join(WORKSPACE, "config.json")
MODEL = "minimax-m2.5:cloud"
REVIEWER_MODEL = "minimax-m2.5:cloud"
MAX_TURNS = 5

os.makedirs(WORKSPACE, exist_ok=True)


# ─── Permission Mode (from claw-code-main pattern) ───

class PermissionMode(IntEnum):
    READ_ONLY       = 0
    WORKSPACE_WRITE = 1
    DANGER_FULL     = 2
    PROMPT          = 3

    @classmethod
    def from_str(cls, s: str) -> "PermissionMode":
        mapping = {
            "read-only": cls.READ_ONLY, "readonly": cls.READ_ONLY,
            "workspace-write": cls.WORKSPACE_WRITE, "write": cls.WORKSPACE_WRITE,
            "danger-full-access": cls.DANGER_FULL, "danger": cls.DANGER_FULL,
            "full": cls.DANGER_FULL, "prompt": cls.PROMPT,
        }
        return mapping.get(s.lower(), cls.WORKSPACE_WRITE)

    def allows(self, required: "PermissionMode") -> bool:
        return self.value >= required.value


# ─── Config File ───

@dataclass
class AgentConfig:
    permission_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE
    workspace_roots: list[str] = field(default_factory=lambda: [os.getcwd()])
    denied_tools: list[str] = field(default_factory=lambda: [
        "rm", "dd", "mkfs", "fdisk", "format", "shutdown", "reboot",
        "poweroff", "init", "killall", "pkill",
    ])
    denied_prefixes: list[str] = field(default_factory=lambda: [
        "sudo ", "su ", "chmod 777", "chown ", "passwd",
    ])
    max_read_size: int = 10 * 1024 * 1024
    max_write_size: int = 10 * 1024 * 1024
    shell_timeout: int = 60

    @classmethod
    def load(cls) -> "AgentConfig":
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
                cfg = cls()
                if "permission_mode" in data:
                    cfg.permission_mode = PermissionMode.from_str(data["permission_mode"])
                if "workspace_roots" in data:
                    cfg.workspace_roots = [os.path.abspath(p) for p in data["workspace_roots"]]
                if "denied_tools" in data:
                    cfg.denied_tools = data["denied_tools"]
                if "denied_prefixes" in data:
                    cfg.denied_prefixes = data["denied_prefixes"]
                if "shell_timeout" in data:
                    cfg.shell_timeout = data["shell_timeout"]
                return cfg
            except Exception as e:
                print(f"[config] load error: {e}, using defaults")
        return cls()

    def save(self):
        data = {
            "permission_mode": self.permission_mode.name.lower(),
            "workspace_roots": self.workspace_roots,
            "denied_tools": self.denied_tools,
            "denied_prefixes": self.denied_prefixes,
            "shell_timeout": self.shell_timeout,
        }
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[config] save error: {e}")


# ─── Path Scope Validation (from claw-code-main path_scope.py) ───

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_UNC_RE   = re.compile(r"^(?:\\\\|//)[^\\/]+[\\/][^\\/]+")
_GLOB_META        = set("*?[")
_ENV_ASSIGN_RE    = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(frozen=True)
class ScopeDecision:
    allowed: bool
    reason: str
    candidate: str | None = None
    resolved: str | None = None


class WorkspaceScope:
    def __init__(self, roots: list[str]):
        self.roots = tuple(Path(r).expanduser().resolve(strict=False) for r in roots)
        if not self.roots:
            self.roots = (Path.cwd().resolve(),)

    def validate(self, payload: str, cwd: str | None = None) -> ScopeDecision:
        cwd_path = Path(cwd).expanduser().resolve(strict=False) if cwd else self.roots[0]
        cwd_check = self._check_path(cwd_path)
        if not cwd_check.allowed:
            return ScopeDecision(False, f"cwd outside workspace: {cwd_path}", str(cwd_path), cwd_check.resolved)
        for candidate in self._extract_candidates(payload):
            decision = self._check_path(self._resolve(candidate, cwd_path))
            if not decision.allowed:
                return decision
        return ScopeDecision(True, "all paths inside workspace")

    def _resolve(self, raw: str, cwd: Path) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        p = Path(expanded)
        if not p.is_absolute():
            p = cwd / p
        return p

    def _check_path(self, path: Path) -> ScopeDecision:
        expanded = self._expand_glob(path)
        for ep in expanded:
            try:
                resolved = ep.resolve(strict=False)
            except OSError:
                resolved = ep.absolute()
            if not any(self._is_relative_to(resolved, root) for root in self.roots):
                return ScopeDecision(False, "path outside workspace", str(path), str(resolved))
        return ScopeDecision(True, "allowed", str(path))

    def _expand_glob(self, path: Path) -> list[Path]:
        text = str(path)
        if any(c in text for c in _GLOB_META):
            matches = [Path(m) for m in glob_mod.glob(text, recursive=True)]
            if matches:
                return matches
            stable = []
            for part in path.parts:
                if any(c in part for c in _GLOB_META):
                    break
                stable.append(part)
            if stable:
                return [Path(*stable)]
        return [path]

    @staticmethod
    def _extract_candidates(payload: str) -> list[str]:
        try:
            tokens = shlex.split(payload, posix=True)
        except ValueError:
            tokens = payload.split()
        raw_tokens = payload.split()
        candidates: list[str] = []
        for token in (*tokens, *raw_tokens):
            if not token or token.startswith("-") or _ENV_ASSIGN_RE.match(token):
                continue
            expanded = os.path.expandvars(os.path.expanduser(token))
            if WorkspaceScope._looks_like(expanded):
                if expanded not in candidates:
                    candidates.append(expanded)
        return candidates

    @staticmethod
    def _looks_like(token: str) -> bool:
        return (
            token in (".", "..")
            or token.startswith(("./", "../", "/", "~/"))
            or "/" in token
            or "\\" in token
            or any(c in token for c in _GLOB_META)
            or bool(_WINDOWS_DRIVE_RE.match(token))
            or bool(_WINDOWS_UNC_RE.match(token))
        )

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


# ─── Bash Command Classification ───

READ_ONLY_COMMANDS = {
    "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "ag",
    "ls", "find", "stat", "wc", "du", "df", "file", "which", "whereis",
    "echo", "printf", "env", "printenv", "pwd", "date", "cal",
    "diff", "cmp", "comm", "sort", "uniq", "cut", "tr", "fold",
    "od", "xxd", "hexdump", "strings", "nl", "pr",
}

DESTRUCTIVE_PATTERNS = [
    (r"\brm\s+-rf\s*/", "rm -rf on root"),
    (r"\bdd\s+if=", "dd (raw disk write)"),
    (r"\bmkfs\b", "mkfs (format filesystem)"),
    (r"\bfdisk\b", "fdisk (partition table)"),
    (r"\bformat\b", "format"),
    (r"\b:\(\)\s*\{", "fork bomb"),
    (r"\bshutdown\b", "shutdown"),
    (r"\breboot\b", "reboot"),
    (r"\binit\s+0\b", "init 0"),
    (r"\bpoweroff\b", "poweroff"),
]


def classify_command(cmd: str) -> tuple[str, str | None]:
    stripped = cmd.strip()
    has_meta = any(c in stripped for c in ";|&$`><()\n")
    first_word = stripped.split(maxsplit=1)[0] if stripped else ""
    for pat, warning in DESTRUCTIVE_PATTERNS:
        if re.search(pat, stripped):
            return ("destructive", warning)
    if first_word in READ_ONLY_COMMANDS and not has_meta:
        return ("read-only", None)
    if first_word in ("curl", "wget", "nc", "netcat", "ssh", "scp", "sftp", "ftp"):
        return ("network", None)
    if first_word in ("apt", "apt-get", "dpkg", "yum", "dnf", "pacman", "brew", "pip", "npm"):
        return ("package", None)
    return ("write", None)


# ─── Tool Permission Context ───

class ToolPermissionContext:
    def __init__(self, config: AgentConfig, cwd: str | None = None):
        self.mode = config.permission_mode
        self.denied_tools = set(t.lower() for t in config.denied_tools)
        self.denied_prefixes = tuple(p.lower() for p in config.denied_prefixes)
        self.scope = WorkspaceScope(config.workspace_roots)
        self.cwd = cwd or os.getcwd()

    def blocks_tool(self, cmd: str) -> str | None:
        first_word = cmd.strip().split(maxsplit=1)[0].lower() if cmd.strip() else ""
        if first_word in self.denied_tools:
            return f"tool '{first_word}' is in denied list"
        for prefix in self.denied_prefixes:
            if cmd.strip().lower().startswith(prefix):
                return f"command starts with denied prefix '{prefix}'"
        return None

    def check_bash(self, cmd: str) -> tuple[bool, str]:
        classification, warning = classify_command(cmd)
        if self.mode == PermissionMode.READ_ONLY:
            if classification != "read-only":
                return (False, f"read-only mode: {classification} not allowed")
            return (True, "")
        if self.mode == PermissionMode.PROMPT:
            return (True, "prompt")
        if self.mode == PermissionMode.WORKSPACE_WRITE:
            if classification == "destructive":
                return (False, f"destructive blocked: {warning}")
            return (True, "")
        return (True, "")

    def check_file_access(self, path: str, write: bool = False) -> tuple[bool, str]:
        if self.mode == PermissionMode.READ_ONLY and write:
            return (False, "read-only: writes not allowed")
        if write and self.mode == PermissionMode.PROMPT:
            return (True, "prompt")
        decision = self.scope.validate(path, self.cwd)
        if not decision.allowed:
            return (False, decision.reason)
        return (True, "")


# ─── Memory ───

conversation_history = []
key_info = []
outside_access_granted = set()


# ─── Ollama API (synchronous, using stdlib urllib) ───

def _ollama_post(payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"response": f"[error: {e}]"}


def call_ollama(prompt: str, system: str = "") -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    result = _ollama_post({"model": MODEL, "prompt": full_prompt, "stream": False})
    return result.get("response", "").strip()


# ─── Security: LLM-based review (additional layer) ───

def review_command(cmd: str) -> tuple[bool, str]:
    review_prompt = f"""You are a security reviewer. Judge if this shell command is safe.

Safety rules:
1. Reading files, browsing directories, searching code = allowed
2. Running dev tools (git, ls, cat, grep, find, python, node) = allowed
3. Deleting data (rm -rf, dd, mkfs) = forbidden
4. Modifying system (sudo, chmod 777, system config) = forbidden
5. Network download+execute = forbidden
6. Any command that could leak data or damage the system = forbidden

Command: {cmd}

Output exactly:
- SAFE
- UNSAFE - reason

Nothing else."""

    try:
        result = _ollama_post(
            {"model": REVIEWER_MODEL, "prompt": review_prompt, "stream": False},
            timeout=60,
        )
        response = result.get("response", "").strip()
        if response.startswith("SAFE"):
            return True, ""
        reason = response.replace("UNSAFE", "").strip(" -")
        return False, reason
    except Exception as e:
        return False, f"review failed: {e}"


# ─── Outside Access Check (original, kept for compat) ───

def check_outside_access(cmd: str, cwd: str) -> tuple[bool, str]:
    import os.path
    def extract_paths(c):
        paths = []
        patterns = [
            (r"(?:^|\s)(?:cat|ls|cd|rm|cp|mv|chmod|chown|find|grep)\s+(/[^\s]+)", 1),
            (r"(?:^|\s)\.\.[^\s]*", 0),
        ]
        for pattern, group in patterns:
            for match in re.finditer(pattern, c, re.MULTILINE):
                path = match.group(group).strip() if group > 0 else ".."
                if path:
                    paths.append(path)
        return paths
    paths = extract_paths(cmd)
    cwd_abs = os.path.abspath(cwd)
    for path in paths:
        if path == ".." or path.startswith("../"):
            abs_path = os.path.abspath(os.path.join(cwd, path))
            return True, abs_path
        if path.startswith("/"):
            abs_path = path
        else:
            abs_path = os.path.abspath(os.path.join(cwd, path))
        if not abs_path.startswith(cwd_abs):
            return True, abs_path
    return False, ""


# ─── UI Helpers ───

def ask_user(prompt_text: str) -> bool:
    print(f"  {prompt_text} (y/N): ", end="")
    try:
        resp = input().strip().lower()
        return resp in ("y", "yes")
    except:
        return False


def mode_name(mode: PermissionMode) -> str:
    return {
        PermissionMode.READ_ONLY: "read-only",
        PermissionMode.WORKSPACE_WRITE: "workspace-write",
        PermissionMode.DANGER_FULL: "danger-full-access",
        PermissionMode.PROMPT: "interactive-prompt",
    }.get(mode, "unknown")


# ─── Memory Management ───

def build_context():
    parts = []
    if key_info:
        items = "\n".join(f"  <item>{k}</item>" for k in key_info)
        parts.append(f"<memory>\n{items}\n</memory>")
    if conversation_history:
        parts.append("<history>\n" + "\n".join(conversation_history[-MAX_TURNS*2:]) + "\n</history>")
    return "\n\n".join(parts)


def update_memory(user_input, assistant_response, tool_result=None):
    conversation_history.append(f"  <user>{user_input}</user>")
    conversation_history.append(f"  <assistant>{assistant_response}</assistant>")
    if tool_result:
        conversation_history.append(f"  <tool>{tool_result[:500]}</tool>")
    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)


def extract_key_info(user_input, assistant_response):
    prompt = f"""Based on this conversation, is there key info for long-term memory?
If yes, output in this format (max 2 items). If no, output <memory></memory>.

<memory>
  <item>info 1</item>
  <item>info 2</item>
</memory>

Conversation:
<user>{user_input}</user>
<assistant>{assistant_response}</assistant>"""
    try:
        result = call_ollama(prompt, "")
        matches = re.findall(r"<item>(.*?)</item>", result, re.DOTALL)
        for item in matches:
            item = item.strip()
            if item and item not in key_info:
                key_info.append(item)
    except:
        pass


# ─── Agent ───

SYSTEM_PROMPT = """You are Jarvis, a helpful AI assistant with security controls.

Rules:
1. To run a shell command, wrap it in <shell> tags
2. <shell> can contain multi-line commands (use && or \\)
3. When done, output <end/>

Flow:
- Need command  -> output <shell>...</shell>
- Need more     -> output another <shell>
- Finished      -> output <end/>"""


def main():
    config = AgentConfig.load()
    config.save()

    print(f"\nAgent0 - {MODEL}")
    print(f"Permission mode: {mode_name(config.permission_mode)}")
    print(f"Workspace roots: {config.workspace_roots}")
    print(f"Commands: /quit, /memory, /mode, /config\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!"); break

        if not user_input: continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("bye!"); break
        if user_input.lower() == "/memory":
            print(f"key info: {key_info}"); continue
        if user_input.lower() == "/mode":
            print(f"mode: {mode_name(config.permission_mode)}"); continue
        if user_input.lower() == "/config":
            print(json.dumps({
                "permission_mode": mode_name(config.permission_mode),
                "workspace_roots": config.workspace_roots,
                "denied_tools": config.denied_tools,
                "shell_timeout": config.shell_timeout,
            }, indent=2))
            continue

        context = build_context()
        full_prompt = f"{context}\n\n<user>{user_input}</user>" if context else f"<user>{user_input}</user>"

        response = call_ollama(full_prompt, SYSTEM_PROMPT)

        tool_result = None
        current_response = response

        while True:
            if "<end/>" in current_response:
                response = current_response.split("<end/>")[0].strip()
                break

            shell_matches = re.findall(r"<shell>(.+?)</shell>", current_response, re.DOTALL)
            if not shell_matches:
                response = current_response
                break

            perm_ctx = ToolPermissionContext(config, os.getcwd())
            all_outputs = []

            for cmd in shell_matches:
                cmd = cmd.strip()

                # Layer 1: denied tool
                denial = perm_ctx.blocks_tool(cmd)
                if denial:
                    print(f"\n  BLOCKED: {cmd}")
                    print(f"  reason: {denial}")
                    all_outputs.append(f"$ {cmd}\nblocked: {denial}")
                    continue

                # Layer 2: bash validation
                bash_ok, bash_reason = perm_ctx.check_bash(cmd)
                if not bash_ok:
                    print(f"\n  BLOCKED: {cmd}")
                    print(f"  reason: {bash_reason}")
                    all_outputs.append(f"$ {cmd}\nblocked: {bash_reason}")
                    continue

                # Layer 3: interactive prompt
                if bash_reason == "prompt":
                    print(f"\n  needs approval: {cmd}")
                    if not ask_user("allow?"):
                        print("  denied")
                        all_outputs.append(f"$ {cmd}\ndenied")
                        continue

                # Layer 4: workspace scope
                decision = perm_ctx.scope.validate(cmd, os.getcwd())
                if not decision.allowed:
                    if decision.candidate in outside_access_granted:
                        pass
                    elif ask_user(f"allow outside access to {decision.candidate}?"):
                        outside_access_granted.add(decision.candidate)
                    else:
                        print("  denied")
                        all_outputs.append(f"$ {cmd}\nblocked: {decision.reason}")
                        continue

                # Layer 5: LLM review
                if config.permission_mode != PermissionMode.DANGER_FULL:
                    is_safe, reason = review_command(cmd)
                    if not is_safe:
                        print(f"\n  BLOCKED by safety review: {cmd}")
                        print(f"  reason: {reason}")
                        all_outputs.append(f"$ {cmd}\nsafety: {reason}")
                        continue

                # Execute
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True,
                        timeout=config.shell_timeout, cwd=os.getcwd()
                    )
                    output = result.stdout + result.stderr
                    if output:
                        print(f"\n  $ {cmd}")
                        for line in output.rstrip().splitlines()[:30]:
                            print(f"    {line}")
                        if len(output.splitlines()) > 30:
                            print(f"    ... ({len(output.splitlines())-30} more)")
                    else:
                        print(f"\n  $ {cmd}  (no output)")
                    all_outputs.append(f"$ {cmd}\n{output if output else '(no output)'}")
                except subprocess.TimeoutExpired:
                    print(f"\n  TIMEOUT: {cmd}")
                    all_outputs.append(f"$ {cmd}\ntimeout")
                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    all_outputs.append(f"$ {cmd}\nerror: {e}")

            tool_result = (tool_result or "") + "\n" + "\n".join(all_outputs)

            follow = f"""<context>{context}</context>

<user>{user_input}</user>
<assistant>{current_response}</assistant>
<output>
{"\n".join(all_outputs)}
</output>

If you need more commands, output <shell>. Otherwise output <end/>."""
            current_response = call_ollama(follow, SYSTEM_PROMPT)

        print(f"\n{response}\n")
        update_memory(user_input, response, tool_result)
        if tool_result:
            extract_key_info(user_input, response)


if __name__ == "__main__":
    main()
