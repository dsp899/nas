from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional


CPUPolicy = str


@dataclass
class CPUState:
    cpu: int
    cpufreq_dir: str
    governor: Optional[str]
    min_khz: Optional[int]
    max_khz: Optional[int]
    setspeed_khz: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "cpu": self.cpu,
            "cpufreq_dir": self.cpufreq_dir,
            "governor": self.governor,
            "min_khz": self.min_khz,
            "max_khz": self.max_khz,
            "setspeed_khz": self.setspeed_khz,
        }


@dataclass
class TurboState:
    path: Optional[str]
    value: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {"path": self.path, "value": self.value}


@dataclass
class CPUControlSnapshot:
    cpus: List[int]
    available: bool
    cpu_states: List[CPUState]
    turbo: TurboState

    def to_dict(self) -> Dict[str, object]:
        return {
            "cpus": self.cpus,
            "available": self.available,
            "cpu_states": [s.to_dict() for s in self.cpu_states],
            "turbo": self.turbo.to_dict(),
        }


@dataclass
class CPUControlResult:
    policy: str
    requested_freq_khz: Optional[int]
    disable_turbo: bool
    affinity_cpus: List[int]
    before: CPUControlSnapshot
    after_apply: Optional[CPUControlSnapshot]
    restored: bool
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "requested_freq_khz": self.requested_freq_khz,
            "disable_turbo": self.disable_turbo,
            "affinity_cpus": self.affinity_cpus,
            "before": self.before.to_dict(),
            "after_apply": None if self.after_apply is None else self.after_apply.to_dict(),
            "restored": self.restored,
            "notes": list(self.notes),
        }


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _write_text(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def _read_int(path: str) -> Optional[int]:
    value = _read_text(path)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _get_affinity_cpus() -> List[int]:
    try:
        if hasattr(os, "sched_getaffinity"):
            return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception:
        pass
    return [0]


def _cpufreq_dir(cpu: int) -> str:
    return f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"


def _read_cpu_state(cpu: int) -> CPUState:
    base = _cpufreq_dir(cpu)
    return CPUState(
        cpu=cpu,
        cpufreq_dir=base,
        governor=_read_text(os.path.join(base, "scaling_governor")),
        min_khz=_read_int(os.path.join(base, "scaling_min_freq")),
        max_khz=_read_int(os.path.join(base, "scaling_max_freq")),
        setspeed_khz=_read_int(os.path.join(base, "scaling_setspeed")),
    )


def _find_turbo_control() -> TurboState:
    intel_path = "/sys/devices/system/cpu/intel_pstate/no_turbo"
    boost_path = "/sys/devices/system/cpu/cpufreq/boost"
    if os.path.exists(intel_path):
        return TurboState(path=intel_path, value=_read_text(intel_path))
    if os.path.exists(boost_path):
        return TurboState(path=boost_path, value=_read_text(boost_path))
    return TurboState(path=None, value=None)


def collect_cpu_snapshot(cpus: Optional[List[int]] = None) -> CPUControlSnapshot:
    target_cpus = cpus or _get_affinity_cpus()
    states: List[CPUState] = []
    available = True
    for cpu in target_cpus:
        base = _cpufreq_dir(cpu)
        if not os.path.isdir(base):
            available = False
            continue
        states.append(_read_cpu_state(cpu))
    if not states:
        available = False
    return CPUControlSnapshot(
        cpus=list(target_cpus),
        available=available,
        cpu_states=states,
        turbo=_find_turbo_control(),
    )


def _apply_turbo(disable_turbo: bool, notes: List[str]) -> None:
    turbo = _find_turbo_control()
    if turbo.path is None:
        notes.append("no_turbo_control_available")
        return
    try:
        if turbo.path.endswith("intel_pstate/no_turbo"):
            _write_text(turbo.path, "1" if disable_turbo else "0")
        elif turbo.path.endswith("cpufreq/boost"):
            _write_text(turbo.path, "0" if disable_turbo else "1")
    except Exception as exc:
        raise RuntimeError(f"No se pudo configurar turbo en {turbo.path}: {exc}") from exc


def _restore_turbo(snapshot: TurboState, notes: List[str]) -> None:
    if snapshot.path is None or snapshot.value is None:
        return
    try:
        _write_text(snapshot.path, snapshot.value)
    except Exception:
        notes.append(f"restore_turbo_failed:{snapshot.path}")


def _apply_cpu_state(cpu_state: CPUState, target_khz: int, notes: List[str]) -> None:
    base = cpu_state.cpufreq_dir
    gov_path = os.path.join(base, "scaling_governor")
    min_path = os.path.join(base, "scaling_min_freq")
    max_path = os.path.join(base, "scaling_max_freq")
    setspeed_path = os.path.join(base, "scaling_setspeed")

    governors_raw = _read_text(os.path.join(base, "scaling_available_governors")) or ""
    governors = governors_raw.split()

    # Try to switch to userspace first for a fixed frequency, otherwise performance.
    try:
        if "userspace" in governors:
            _write_text(gov_path, "userspace")
            notes.append(f"cpu{cpu_state.cpu}:governor=userspace")
        elif "performance" in governors:
            _write_text(gov_path, "performance")
            notes.append(f"cpu{cpu_state.cpu}:governor=performance")
    except Exception as exc:
        raise RuntimeError(f"No se pudo fijar governor en cpu{cpu_state.cpu}: {exc}") from exc

    try:
        _write_text(min_path, str(target_khz))
        _write_text(max_path, str(target_khz))
    except Exception as exc:
        raise RuntimeError(f"No se pudo fijar min/max freq en cpu{cpu_state.cpu}: {exc}") from exc

    if os.path.exists(setspeed_path):
        try:
            _write_text(setspeed_path, str(target_khz))
        except Exception:
            notes.append(f"cpu{cpu_state.cpu}:setspeed_write_failed")


def _restore_cpu_state(cpu_state: CPUState, notes: List[str]) -> None:
    try:
        if cpu_state.governor is not None:
            _write_text(os.path.join(cpu_state.cpufreq_dir, "scaling_governor"), cpu_state.governor)
    except Exception:
        notes.append(f"restore_governor_failed:cpu{cpu_state.cpu}")
    try:
        if cpu_state.min_khz is not None:
            _write_text(os.path.join(cpu_state.cpufreq_dir, "scaling_min_freq"), str(cpu_state.min_khz))
    except Exception:
        notes.append(f"restore_min_freq_failed:cpu{cpu_state.cpu}")
    try:
        if cpu_state.max_khz is not None:
            _write_text(os.path.join(cpu_state.cpufreq_dir, "scaling_max_freq"), str(cpu_state.max_khz))
    except Exception:
        notes.append(f"restore_max_freq_failed:cpu{cpu_state.cpu}")
    setspeed_path = os.path.join(cpu_state.cpufreq_dir, "scaling_setspeed")
    if cpu_state.setspeed_khz is not None and os.path.exists(setspeed_path):
        try:
            _write_text(setspeed_path, str(cpu_state.setspeed_khz))
        except Exception:
            notes.append(f"restore_setspeed_failed:cpu{cpu_state.cpu}")


@contextmanager
def cpu_policy_scope(policy: str = "none", freq_khz: Optional[int] = None, disable_turbo: bool = False) -> Iterator[CPUControlResult]:
    affinity_cpus = _get_affinity_cpus()
    notes: List[str] = []
    before = collect_cpu_snapshot(affinity_cpus)
    result = CPUControlResult(
        policy=policy,
        requested_freq_khz=freq_khz,
        disable_turbo=disable_turbo,
        affinity_cpus=affinity_cpus,
        before=before,
        after_apply=None,
        restored=False,
        notes=notes,
    )

    if policy == "none":
        notes.append("cpu_policy_none")
        try:
            yield result
        finally:
            result.restored = True
        return

    if policy not in ("validate", "fix"):
        raise ValueError(f"cpu policy no soportada: {policy}")

    if not before.available:
        if policy == "fix":
            raise RuntimeError("No hay interfaz cpufreq disponible para fijar frecuencia de CPU en este host.")
        notes.append("cpufreq_not_available")
        try:
            yield result
        finally:
            result.restored = True
        return

    if policy == "validate":
        notes.append("cpu_policy_validate")
        result.after_apply = collect_cpu_snapshot(affinity_cpus)
        try:
            yield result
        finally:
            result.restored = True
        return

    # policy == fix
    if freq_khz is None:
        raise RuntimeError("cpu_policy=fix requiere --cpu-freq-khz")

    try:
        if disable_turbo:
            _apply_turbo(True, notes)
        for state in before.cpu_states:
            _apply_cpu_state(state, freq_khz, notes)
        result.after_apply = collect_cpu_snapshot(affinity_cpus)
        notes.append("cpu_policy_fix_applied")
        yield result
    finally:
        for state in before.cpu_states:
            _restore_cpu_state(state, notes)
        if disable_turbo:
            _restore_turbo(before.turbo, notes)
        result.restored = True
