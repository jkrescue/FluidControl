import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_resource_lifecycle_preserves_physical_fields_and_exact_thresholds(monkeypatch):
    life = load("p064_h25_life", ROOT / "scripts/run_p064_b_h25_resource_lifecycle.py")
    text = (ROOT / "scripts/run_p064_b_h25_resource_lifecycle.py").read_text()
    assert 'memory["MemAvailable"] >= 80 * GIB' in text
    assert 'row["MemAvailable"] >= 22 * GIB' in text
    assert 'min(row["MemFree"], row["MemAvailable"])' not in text
    monkeypatch.setattr(life.Path, "read_text", lambda self: "MemFree: 11 kB\nMemAvailable: 23 kB\n")
    assert life.host_memory() == {"MemFree": 11 * 1024, "MemAvailable": 23 * 1024}


def test_created_container_requires_48g_no_swap(tmp_path):
    life = load("p064_h25_life_created", ROOT / "scripts/run_p064_b_h25_resource_lifecycle.py")
    cid = "a" * 64
    inspect = {
        "Id": cid, "Image": life.IMAGE, "Name": "/" + life.CONTAINER,
        "HostConfig": {"Memory": 48 * life.GIB, "MemorySwap": 48 * life.GIB,
                       "NetworkMode": "none", "ReadonlyRootfs": True,
                       "DeviceRequests": [{"Capabilities": [["gpu"]], "DeviceIDs": ["0"]}]},
        "Mounts": [{"Type": "bind", "Source": str(tmp_path.resolve()),
                    "Destination": str(tmp_path.resolve()), "RW": True}],
        "Config": {"Cmd": ["true"]},
    }
    life.validate_created(inspect, cid, tmp_path, [], ["true"])
    inspect["HostConfig"]["Memory"] = 12 * life.GIB
    try:
        life.validate_created(inspect, cid, tmp_path, [], ["true"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("12 GiB container was accepted")


def test_launcher_modes_and_no_inner_legacy_guard():
    launcher = load("p064_h25_launcher", ROOT / "scripts/run_p064_b_flow_h25_bounded.py")
    assert set(launcher.PROFILES) == {"scales", "resource-probe", "train"}
    assert launcher.PROFILES["scales"]["training_windows"] == 1368
    assert launcher.PROFILES["resource-probe"]["optimizer_steps"] == 1
    assert launcher.PROFILES["train"]["optimizer_steps"] == 32
    text = (ROOT / "scripts/run_p064_b_flow_h25_bounded.py").read_text()
    assert 'base.host_memory =' not in text
    assert '"--memory", "48g", "--memory-swap", "48g"' in text
