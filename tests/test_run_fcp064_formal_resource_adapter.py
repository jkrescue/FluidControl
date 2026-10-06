import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_fcp064_formal_resource_adapter.py"
spec = importlib.util.spec_from_file_location("p064_resource", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Base:
    IMAGE = "image"
    PREDECL = "predecl"
    QC = "qc"
    DATA = {"devdata": "dev", "dynamic": "dyn", "full40": "full"}

    @staticmethod
    def require(value, message):
        if not value:
            raise RuntimeError(message)


class Process:
    returncode = 0

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


class ResourceAdapterTest(unittest.TestCase):
    def test_explicit_resource_contract_and_actual_memory_fields(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, candidate, output = root / "source", root / "candidate", root / "output"
            for path in (source, candidate, output / "evidence", root / "dev", root / "dyn", root / "full"):
                path.mkdir(parents=True)
            (root / "predecl").write_text("x")
            (root / "qc").write_text("x")
            args = type("Args", (), {"source": source, "candidate": candidate, "output": output, "repo": root})()
            created = {"Image": "image", "Mounts": [{"Source": str(output), "Destination": "/workspace/output"}],
                       "HostConfig": {"Memory": 72 * module.GIB, "MemorySwap": 72 * module.GIB, "PidsLimit": 1024}}
            terminal = {**created, "State": {"ExitCode": 0, "OOMKilled": False, "Running": False}}
            calls = []

            def check_output(command, **_):
                calls.append(command)
                if command[:2] == ["docker", "create"]:
                    return "a" * 64 + "\n"
                return json.dumps([created if len(calls) == 2 else terminal])

            memory = {"MemFree": 1 * module.GIB, "MemAvailable": 60 * module.GIB}
            with mock.patch.object(module.subprocess, "check_output", side_effect=check_output), \
                 mock.patch.object(module.subprocess, "Popen", return_value=Process()), \
                 mock.patch.object(module.subprocess, "run"), \
                 mock.patch.object(module, "available_memory", return_value=memory):
                module.adapted_run_container(Base(), args, {}, "validation10", True,
                                             ["python", "worker.py"], time.monotonic() + 10)
            create = calls[0]
            self.assertIn("72g", create)
            self.assertEqual(create[create.index("--memory-swap") + 1], "72g")
            self.assertEqual(create[create.index("--pids-limit") + 1], "1024")
            self.assertIn("--allocator-fraction", create)
            self.assertEqual(create[create.index("--allocator-fraction") + 1], ".06")
            row = json.loads((output / "memory.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["MemFree"], module.GIB)
            self.assertEqual(row["MemAvailable"], 60 * module.GIB)

    def test_runtime_uses_available_not_free(self):
        self.assertGreater(module.RUNTIME_AVAILABLE, 20 * module.GIB)
        self.assertEqual(module.STARTUP_AVAILABLE, 50 * module.GIB)


if __name__ == "__main__":
    unittest.main()
