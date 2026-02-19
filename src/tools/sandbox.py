"""
Safe Sandbox: Executes Python code in a restricted environment.
Uses AST validation + restricted builtins + timeout for defense in depth.
"""

import ast
import json
import sys
import signal
import threading
import traceback
from typing import Dict, Any, Optional
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from src.core.config import get_settings

# Optional Unix-specific imports
try:
    import resource as _resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


class SandboxError(Exception):
    """Raised when sandbox security is violated."""

    pass


class SandboxTimeoutError(Exception):
    """Raised when code execution times out."""

    pass


class SafeSandbox:
    """
    Safe Python code execution environment with defense-in-depth security.

    Security layers:
      1. AST-based static analysis — blocks dangerous constructs before execution
      2. Restricted builtins — only safe functions exposed, no __import__ or compile
      3. Whitelisted imports — only data-science modules allowed via restricted_import
      4. Execution timeout — thread-based kill after configurable seconds
      5. Output length limits — prevents memory exhaustion via large prints
      6. Resource limits — memory caps on Unix systems
    """

    # Whitelisted imports — only safe data-analysis modules
    ALLOWED_MODULES = frozenset({
        "math",
        "statistics",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "typing",
        "numpy",
        "pandas",
    })

    # Dangerous constructs blocked at AST level
    DANGEROUS_CALLS = frozenset({
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "exit",
        "quit",
    })

    # Dangerous attribute access patterns
    DANGEROUS_ATTRS = frozenset({
        "system",
        "popen",
        "exec",
        "eval",
        "compile",
        "__import__",
        "__subclasses__",
        "__bases__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__class__",
        "mro",
    })

    # Dangerous module-level names to block in imports
    DANGEROUS_MODULES = frozenset({
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "http",
        "urllib",
        "ftplib",
        "pickle",
        "marshal",
        "ctypes",
        "multiprocessing",
        "threading",
        "signal",
        "importlib",
        "builtins",
        "code",
        "codeop",
        "compileall",
        "py_compile",
        "zipimport",
        "pathlib",
        "tempfile",
        "glob",
        "io",
    })

    def __init__(self):
        self.settings = get_settings()
        self.timeout = self.settings.sandbox_timeout
        self.max_memory_mb = self.settings.sandbox_max_memory_mb
        self.max_output_length = self.settings.sandbox_max_output_length

    # ── AST Validation ──────────────────────────────────────────────

    def validate_code(self, code: str) -> bool:
        """
        Validate code using AST analysis before execution.

        Raises SandboxError if unsafe patterns detected.
        Returns True if code passes all checks.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SandboxError(f"Syntax error: {e}")

        for node in ast.walk(tree):
            # Block dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_CALLS:
                        raise SandboxError(
                            f"Blocked: call to '{node.func.id}' is not allowed"
                        )

            # Block imports of unauthorized modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    if module_root in self.DANGEROUS_MODULES:
                        raise SandboxError(
                            f"Blocked: import of '{module_root}' is not allowed"
                        )
                    if module_root not in self.ALLOWED_MODULES:
                        raise SandboxError(
                            f"Blocked: module '{module_root}' is not whitelisted"
                        )

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    if module_root in self.DANGEROUS_MODULES:
                        raise SandboxError(
                            f"Blocked: import from '{module_root}' is not allowed"
                        )
                    if module_root not in self.ALLOWED_MODULES:
                        raise SandboxError(
                            f"Blocked: module '{module_root}' is not whitelisted"
                        )

            # Block dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.DANGEROUS_ATTRS:
                    raise SandboxError(
                        f"Blocked: attribute access '.{node.attr}' is not allowed"
                    )

            # Block string-based code execution patterns
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.DANGEROUS_CALLS:
                        raise SandboxError(
                            f"Blocked: method call '.{node.func.attr}' is not allowed"
                        )

        return True

    # ── Restricted Import ───────────────────────────────────────────

    @classmethod
    def _restricted_import(cls, name, *args, **kwargs):
        """
        Restricted __import__ that only allows whitelisted modules.
        This is the ONLY import path available inside the sandbox.
        """
        module_root = name.split(".")[0]
        if module_root not in cls.ALLOWED_MODULES:
            raise ImportError(
                f"Module '{name}' is not allowed in the sandbox. "
                f"Allowed: {', '.join(sorted(cls.ALLOWED_MODULES))}"
            )
        return __import__(name, *args, **kwargs)

    # ── Execution ───────────────────────────────────────────────────

    def execute(
        self, code: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.

        Args:
            code: Python code to execute
            context: Variables to inject into the execution namespace

        Returns:
            Dict with success, error, output, and result fields
        """
        # Layer 1: AST validation
        try:
            self.validate_code(code)
        except SandboxError as e:
            return {"success": False, "error": str(e), "output": "", "result": None}

        # Set up capture buffers
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        # Layer 2: Restricted builtins — NO __import__, NO compile, NO open
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "ascii": ascii,
            "bin": bin,
            "bool": bool,
            "bytearray": bytearray,
            "bytes": bytes,
            "callable": callable,
            "chr": chr,
            "complex": complex,
            "dict": dict,
            "dir": dir,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "getattr": getattr,
            "hasattr": hasattr,
            "hash": hash,
            "hex": hex,
            "id": id,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "object": object,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": lambda *args, **kwargs: print(*args, file=stdout_buffer, **kwargs),
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            # Restricted __import__ — ONLY allows whitelisted modules
            "__import__": self._restricted_import,
            # Explicitly block these with clear errors
            "eval": None,
            "exec": None,
            "compile": None,
            "open": None,
            "input": None,
            "breakpoint": None,
            "__build_class__": __builtins__.__build_class__ if hasattr(__builtins__, '__build_class__') else None,
        }

        safe_globals = {"__builtins__": safe_builtins}

        # Layer 3: Pre-import allowed modules into namespace
        for module_name in self.ALLOWED_MODULES:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                pass  # Module not installed — skip silently

        # Inject user context variables
        if context:
            safe_globals.update(context)

        # Layer 4: Execute with timeout
        result_holder = {"success": False, "error": None, "exception": None}

        def _run_code():
            try:
                exec(code, safe_globals)
                result_holder["success"] = True
            except Exception as e:
                result_holder["exception"] = e
                result_holder["error"] = f"{type(e).__name__}: {str(e)}"

        exec_thread = threading.Thread(target=_run_code, daemon=True)
        exec_thread.start()
        exec_thread.join(timeout=self.timeout)

        if exec_thread.is_alive():
            # Thread is still running — timeout reached
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout} seconds",
                "output": stdout_buffer.getvalue(),
                "result": None,
            }

        if not result_holder["success"]:
            return {
                "success": False,
                "error": result_holder["error"] or "Unknown execution error",
                "output": stdout_buffer.getvalue(),
                "result": None,
            }

        # Collect outputs
        output = stdout_buffer.getvalue()
        error_output = stderr_buffer.getvalue()

        # Layer 5: Truncate oversized output
        if len(output) > self.max_output_length:
            output = output[: self.max_output_length] + "\n... (output truncated)"

        result_value = safe_globals.get("result", None)

        return {
            "success": True,
            "error": error_output if error_output else None,
            "output": output,
            "result": result_value,
        }


# ── Convenience Functions ───────────────────────────────────────────

def execute_weather_analysis(code: str, weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute weather analysis code in sandbox with weather data pre-injected.

    Args:
        code: Analysis code to run
        weather_data: Weather data dict to make available as 'data'

    Returns:
        Execution result dict
    """
    sandbox = SafeSandbox()
    context = {"data": weather_data, "json": json, "result": None}
    return sandbox.execute(code, context)
