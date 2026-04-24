"""
scripts/validate_project.py
----------------------------
Lightweight smoke test: import all package modules and verify data files load.
Does NOT run any optimization or generate results.

Exit code 0 = all checks passed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

CHECKS = []


def check(name: str):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("septa_worldcup package")
def _():
    import septa_worldcup
    assert (septa_worldcup.REPO_ROOT / "data").exists(), "DATA_DIR not found"


@check("v1.data.network")
def _():
    from septa_worldcup.v1.data.network import LINES
    assert len(LINES) == 13, f"Expected 13 lines, got {len(LINES)}"


@check("v1.data.parameters")
def _():
    from septa_worldcup.v1.data.parameters import N_SLOTS, FIXED_COST_PER_TRAIN, slot_label, time_to_slot
    assert N_SLOTS == 40, f"Expected 40 slots, got {N_SLOTS}"
    assert FIXED_COST_PER_TRAIN > 0
    assert slot_label(0) == "18:00"
    assert slot_label(39) == "03:45 (+1)"
    assert time_to_slot("20:30") == 10


@check("v1.data.demand")
def _():
    from septa_worldcup.v1.data.demand import get_total_demand
    d = get_total_demand(worldcup=False)
    assert len(d) > 0


@check("v1.models.lower_level")
def _():
    from septa_worldcup.v1.models.lower_level import effective_demand  # noqa: F401


@check("v1.models.upper_level")
def _():
    from septa_worldcup.v1.models.upper_level import LNAMES, T, TBLOCKS
    assert T == 40, f"Expected T=40, got T={T}"
    assert len(LNAMES) == 13
    assert "pre_game" in TBLOCKS


@check("v2.config.scenario")
def _():
    from septa_worldcup.v2.config.scenario import N_SLOTS, SLOT_MINUTES
    assert N_SLOTS == 40
    assert SLOT_MINUTES == 15


@check("v2.data.worldcup_demand")
def _():
    from septa_worldcup.v2.data.worldcup_demand import get_demand
    d = get_demand()
    assert "rr_demand" in d
    assert "bsl_inbound" in d


@check("v2.data.bsl")
def _():
    from septa_worldcup.v2.data.bsl import allocate_bsl_service  # noqa: F401


@check("v2.models.policy_objective")
def _():
    from septa_worldcup.v2.models.policy_objective import evaluate_rr_service  # noqa: F401


@check("v2.reporting.reporting")
def _():
    from septa_worldcup.v2.reporting.reporting import compute_kpis  # noqa: F401


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} checks passed.")
    sys.exit(0 if failed == 0 else 1)
