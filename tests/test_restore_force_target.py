"""Snapshot restore must restore the force-measurement target, including 'no target' (step 23)."""
import synthesis_v2.grasp_force_contract as gfc
from synthesis_v2.native_osc_env import NativeSession


class _S(NativeSession):
    def __init__(self):  # no simulator
        self.task, self._hooked = "libero", True
        self._binding, self.force_target_name = ("bind", "default"), None
        self.geoms = {"default": (1,), "A": (2,), "B": (3,)}

    def _mj(self):
        return None, None

    def _target_geoms(self):
        return "default", self.geoms["default"]

    def geom_ids_of(self, n):
        return self.geoms[n]


def _patch(monkeypatch):
    monkeypatch.setattr(gfc, "resolve_grasp_binding", lambda m, mj, ids, name: ("bind", name))


def test_target_to_no_target_clears_stale_label_and_binding(monkeypatch):
    _patch(monkeypatch)
    s = _S()
    s.set_force_target("A")
    assert (s.force_target_name, s._binding) == ("A", ("bind", "A"))
    s._restore_force_target({"force_target": None})
    assert s.force_target_name is None and s._binding == ("bind", "default")
    s.set_force_target("B")
    s._restore_force_target({"force_target": ""})
    assert s.force_target_name is None and s._binding == ("bind", "default")


def test_object_a_to_object_b(monkeypatch):
    _patch(monkeypatch)
    s = _S()
    s.set_force_target("A")
    s._restore_force_target({"force_target": "B"})
    assert (s.force_target_name, s._binding) == ("B", ("bind", "B"))


def test_no_target_to_target_and_legacy_snapshot(monkeypatch):
    _patch(monkeypatch)
    s = _S()
    s._restore_force_target({"force_target": "A"})
    assert s.force_target_name == "A"
    s._restore_force_target({})  # snapshot without the key: untouched
    assert s.force_target_name == "A" and s._binding == ("bind", "A")


def test_restore_before_hook_only_clears_label(monkeypatch):
    _patch(monkeypatch)
    s = _S()
    s._hooked, s._binding, s.force_target_name = False, None, "A"
    s._restore_force_target({"force_target": None})
    assert s.force_target_name is None and s._binding is None
