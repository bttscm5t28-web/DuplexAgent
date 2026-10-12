"""Sandboxed Python execution for the slow brain's `python` tool.

Model-written code never runs in the engine process. Each call is a fresh child process with
layered restrictions (each layer is best-effort; together they make escape hard):

1. `python -I -S`: isolated mode, no site-packages path tricks, no user env vars.
2. Empty environment, a throwaway working directory, stdin carries the code.
3. rlimits: CPU seconds, address space, max file size, open files, no core dumps.
4. If running as root: drop to uid/gid `nobody` (65534) before exec.
5. If `unshare -n` works: run inside a fresh network namespace (no network at all).
6. A PEP 578 audit hook inside the child denies sockets, subprocesses, exec/fork, ctypes,
   file writes, and reads outside the Python install, and blocks importing dangerous modules.
7. Wall-clock timeout; the whole process group is killed on timeout *and* on cancellation.

Not a security boundary for hostile multi-tenant use (use a container/gVisor/nsjail for that),
but it keeps an LLM's code from touching the host, the network, or the engine.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import tempfile

# Runs in the child before the user code. Kept self-contained (only builtins + sys).
_BOOTSTRAP = r'''
import sys, io
_code = sys.stdin.read()
# Pre-import the allowed toolset while imports are still unrestricted.
import math, cmath, statistics, fractions, decimal, datetime, calendar, itertools, functools
import collections, re, json, random, string, heapq, bisect, operator, textwrap, unicodedata
import ast as _ast
_allowed_roots = tuple(p for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix} if p)
_allowed_roots += tuple(p for p in sys.path if p)
_BLOCK_MODS = {"socket", "_socket", "ssl", "subprocess", "_posixsubprocess", "multiprocessing", "ctypes",
               "_ctypes", "pty", "shutil", "urllib", "http", "ftplib", "smtplib", "asyncio", "threading",
               "signal", "resource", "importlib", "pathlib", "tempfile", "glob", "mmap", "fcntl"}
_BLOCK_EVENTS = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                 "os.forkpty", "os.kill", "os.killpg", "os.remove", "os.unlink", "os.rename", "os.rmdir",
                 "os.mkdir", "os.chmod", "os.chown", "os.symlink", "os.link", "os.truncate", "os.putenv",
                 "os.unsetenv", "os.chdir", "os.listdir", "os.scandir", "os.walk", "ctypes.", "sys.addaudithook",
                 "sys.setprofile", "sys.settrace", "sys._getframe", "pickle.", "shutil.", "webbrowser.")
_armed = False
def _hook(event, args):
    if not _armed:
        return
    if event.startswith(_BLOCK_EVENTS):
        raise PermissionError(f"sandbox: {event} is not allowed")
    if event == "import" and args and str(args[0]).split(".")[0] in _BLOCK_MODS:
        raise PermissionError(f"sandbox: import {args[0]} is not allowed")
    if event == "open":
        path, mode = args[0], (args[1] or "r")
        if isinstance(path, int):
            raise PermissionError("sandbox: opening file descriptors is not allowed")
        if any(c in str(mode) for c in "wax+"):
            raise PermissionError("sandbox: writing files is not allowed")
        p = str(path)
        if not p.startswith(_allowed_roots):
            raise PermissionError(f"sandbox: reading {p} is not allowed")
sys.addaudithook(_hook)
_armed = True
_g = {"__name__": "__main__", "__builtins__": __builtins__}
for _m in ("math", "cmath", "statistics", "fractions", "decimal", "datetime", "calendar", "itertools",
           "functools", "collections", "re", "json", "random", "string", "heapq", "bisect", "operator"):
    _g[_m] = sys.modules[_m]
# Auto-print the value of a trailing expression, like a REPL.
try:
    _tree = _ast.parse(_code, "<sandbox>", "exec")
except SyntaxError as e:
    print(f"SyntaxError: {e}", file=sys.stderr); sys.exit(1)
_last = None
if _tree.body and isinstance(_tree.body[-1], _ast.Expr):
    _last = _ast.Expression(_tree.body.pop().value)
exec(compile(_tree, "<sandbox>", "exec"), _g)
if _last is not None:
    _v = eval(compile(_last, "<sandbox>", "eval"), _g)
    if _v is not None:
        print(repr(_v))
'''

_NOBODY = 65534


def _limits(cpu_s: int, mem_mb: int, drop_privileges: bool):
    def apply() -> None:
        import resource

        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if drop_privileges and os.geteuid() == 0:
            os.setgroups([])
            os.setgid(_NOBODY)
            os.setuid(_NOBODY)

    return apply


_netns_ok: bool | None = None


def _netns_available() -> bool:
    global _netns_ok
    if _netns_ok is None:
        exe = shutil.which("unshare")
        _netns_ok = False
        if exe:
            try:
                _netns_ok = subprocess.run([exe, "-n", "true"], capture_output=True, timeout=5).returncode == 0
            except Exception:
                _netns_ok = False
    return _netns_ok


class PythonSandbox:
    def __init__(self, timeout_s: float = 10.0, cpu_s: int = 5, mem_mb: int = 512, max_output: int = 4000, isolate_network: bool = True):
        self.timeout_s = timeout_s
        self.cpu_s = cpu_s
        self.mem_mb = mem_mb
        self.max_output = max_output
        self.isolate_network = isolate_network

    async def run(self, code: str) -> tuple[bool, str]:
        """Run `code`; return (ok, output). Output is stdout (+stderr on failure), truncated."""
        argv = [sys.executable, "-I", "-S", "-c", _BOOTSTRAP]
        drop_in_preexec = True
        if self.isolate_network and _netns_available():
            # unshare needs root, so drop privileges after it with setpriv instead of in preexec.
            setpriv = shutil.which("setpriv")
            if os.geteuid() == 0 and setpriv:
                argv = [setpriv, f"--reuid={_NOBODY}", f"--regid={_NOBODY}", "--clear-groups", "--", *argv]
                drop_in_preexec = False
            if not drop_in_preexec or os.geteuid() != 0:
                argv = [shutil.which("unshare") or "unshare", "-n", "--", *argv]
        workdir = tempfile.mkdtemp(prefix="slowbox-")
        os.chmod(workdir, 0o755)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"},
            preexec_fn=_limits(self.cpu_s, self.mem_mb, drop_privileges=drop_in_preexec),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(code.encode()), self.timeout_s)
        except asyncio.TimeoutError:
            return False, f"超时：代码运行超过 {self.timeout_s:g} 秒被终止"
        finally:  # timeout or task cancellation: never leave the child running
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
            shutil.rmtree(workdir, ignore_errors=True)
        so, se = out.decode(errors="replace"), err.decode(errors="replace")
        if proc.returncode == 0:
            text = so.strip() or "(没有输出；请用 print 打印结果)"
            return True, self._clip(text)
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            return False, "超出 CPU 时间限制被终止"
        lines = [ln for ln in se.strip().splitlines() if not ln.lstrip().startswith(("File ", "Traceback", "^"))]
        tail = lines[-2:] if lines else [f"exit code {proc.returncode}"]
        return False, self._clip((so.strip() + "\n" if so.strip() else "") + "\n".join(tail))

    def _clip(self, s: str) -> str:
        return s if len(s) <= self.max_output else s[: self.max_output] + f"\n…（截断，共 {len(s)} 字符）"
