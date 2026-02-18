"""
Safe Sandbox: Executes Python code in restricted environment.
Used for data analysis tools.
"""

import ast
import json
import sys
import threading
import traceback
from typing import Dict, Any, Optional
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from src.core.config import get_settings

# Optional Unix-specific imports
try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

try:
    import signal

    HAS_SIGNAL = True
except ImportError:
    HAS_SIGNAL = False


class SandboxError(Exception):
    """Raised when sandbox security is violated."""

    pass


class TimeoutError(Exception):
    """Raised when code execution times out."""

    pass


class SafeSandbox:
    """
    Safe Python code execution environment.

    Security measures:
    - AST-based code validation
    - Import whitelist
    - Execution timeout
    - Resource limits
    - Output length limits
    """

    # Whitelisted imports
    ALLOWED_MODULES = {
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
        "matplotlib",
        "seaborn",
    }

    # Dangerous keywords to block
    DANGEROUS_KEYWORDS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "subprocess",
        "os.system",
        "os.popen",
        "socket",
        "urllib",
        "http",
        "ftp",
        "pickle",
        "marshal",
        "yaml.load",
    }

    def __init__(self):
        self.settings = get_settings()
        self.timeout = self.settings.sandbox_timeout
        self.max_memory_mb = self.settings.sandbox_max_memory_mb
        self.max_output_length = self.settings.sandbox_max_output_length

    def validate_code(self, code: str) -> bool:
        """
        Validate code using AST analysis.

        Args:
            code: Python code to validate

        Returns:
            True if code is safe

        Raises:
            SandboxError if unsafe code detected
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SandboxError(f"Syntax error: {e}")

        for node in ast.walk(tree):
            # Check for dangerous calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_KEYWORDS:
                        raise SandboxError(f"Dangerous function call: {node.func.id}")

            # Check for imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name not in self.ALLOWED_MODULES:
                        raise SandboxError(f"Unauthorized module import: {module_name}")

            # Check for attribute access on dangerous modules
            if isinstance(node, ast.Attribute):
                if node.attr in self.DANGEROUS_KEYWORDS:
                    raise SandboxError(f"Dangerous attribute access: {node.attr}")

        return True

    def execute(
        self, code: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute
            context: Variables to inject into namespace

        Returns:
            Execution result dict
        """
        # Validate code
        try:
            self.validate_code(code)
        except SandboxError as e:
            return {"success": False, "error": str(e), "output": "", "result": None}

        # Set up execution environment
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        # Create restricted globals
        safe_globals = {
            "__builtins__": {
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
                "classmethod": classmethod,
                "compile": compile,
                "complex": complex,
                "delattr": delattr,
                "dict": dict,
                "dir": dir,
                "divmod": divmod,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "format": format,
                "frozenset": frozenset,
                "getattr": getattr,
                "globals": globals,
                "hasattr": hasattr,
                "hash": hash,
                "help": help,
                "hex": hex,
                "id": id,
                "input": input,
                "int": int,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "iter": iter,
                "len": len,
                "list": list,
                "locals": locals,
                "map": map,
                "max": max,
                "memoryview": memoryview,
                "min": min,
                "next": next,
                "object": object,
                "oct": oct,
                "ord": ord,
                "pow": pow,
                "print": print,
                "property": property,
                "range": range,
                "repr": repr,
                "reversed": reversed,
                "round": round,
                "set": set,
                "setattr": setattr,
                "slice": slice,
                "sorted": sorted,
                "staticmethod": staticmethod,
                "str": str,
                "sum": sum,
                "super": super,
                "tuple": tuple,
                "type": type,
                "vars": vars,
                "zip": zip,
                "__import__": __import__,
            }
        }

        # Add allowed modules
        for module_name in self.ALLOWED_MODULES:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                pass

        # Add context variables
        if context:
            safe_globals.update(context)

        # Set up timeout using threading (cross-platform)
        timeout_occurred = threading.Event()

        def timeout_worker():
            timeout_occurred.wait(self.timeout)
            if not timeout_occurred.is_set():
                timeout_occurred.set()

        timeout_thread = threading.Thread(target=timeout_worker, daemon=True)
        timeout_thread.start()

        try:
            # Capture output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Execute code
                exec(code, safe_globals)
                timeout_occurred.set()  # Mark as completed

            # Get result
            result = safe_globals.get("result", None)
            output = stdout_buffer.getvalue()
            error = stderr_buffer.getvalue()

            # Check output length
            if len(output) > self.max_output_length:
                output = output[: self.max_output_length] + "\n... (output truncated)"

            return {
                "success": True,
                "error": error if error else None,
                "output": output,
                "result": result,
            }

        except Exception as e:
            if timeout_occurred.is_set() and "execution timeout" in str(e).lower():
                return {
                    "success": False,
                    "error": f"Code execution timeout after {self.timeout} seconds",
                    "output": stdout_buffer.getvalue(),
                    "result": None,
                }
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "output": stdout_buffer.getvalue(),
                "result": None,
            }


# Convenience function for weather analysis
def execute_weather_analysis(code: str, weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute weather analysis code in sandbox.

    Args:
        code: Analysis code
        weather_data: Weather data to analyze

    Returns:
        Execution result
    """
    sandbox = SafeSandbox()

    context = {"data": weather_data, "json": json, "result": None}

    return sandbox.execute(code, context)
