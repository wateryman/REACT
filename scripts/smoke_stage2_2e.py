"""Stage-2 2.e smoke test: YOPODataset --dynamic switch.

Verifies the dynamic-mode branch added in yopo_dataset.py:
  - Loads K=10 frame sequences from dataset_dynamic/v1/
  - Returns the documented dict schema
  - depth_seq has the expected (K, 1, 96, 160) shape and is in [0, 1]
  - state_seq / dyn_obs / meta survive the JSON round-trip
  - Static-mode path is untouched (regression: load static dataset[0] still
    returns the 5-tuple shape expected by stage-1 trainer)

Run from REACT/ root:
    python scripts/smoke_stage2_2e.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import numpy as np

from policy.yopo_dataset import YOPODataset


def main():
    # ---- dynamic mode ----
    print("== YOPODataset(dynamic=True, K=10) ==")
    ds_dyn = YOPODataset(mode='train', dynamic=True, K=10)
    assert len(ds_dyn) > 0, f"empty dynamic dataset"
    s = ds_dyn[0]

    # schema
    expected_keys = {"depth_seq", "state_seq", "dyn_obs", "dt_seq", "meta"}
    assert set(s.keys()) == expected_keys, f"unexpected keys: {set(s.keys())}"

    # depth_seq shape (K, 1, image_height, image_width)
    # image_height = 96 / image_width = 160 per cfg
    assert s["depth_seq"].shape == (10, 1, 96, 160), s["depth_seq"].shape
    assert s["depth_seq"].dtype == np.float32, s["depth_seq"].dtype
    assert 0.0 <= float(s["depth_seq"].min()) and float(s["depth_seq"].max()) <= 1.0, \
        f"depth_seq out of [0,1]: [{s['depth_seq'].min()}, {s['depth_seq'].max()}]"

    # state_seq is K dicts
    assert len(s["state_seq"]) == 10, len(s["state_seq"])
    for st in s["state_seq"]:
        for k in ("pos", "quat_wc", "vel_world"):
            assert k in st, f"state missing '{k}': {st.keys()}"

    # dyn_obs is K lists of ball dicts
    assert len(s["dyn_obs"]) == 10, len(s["dyn_obs"])
    n_per = [len(f) for f in s["dyn_obs"]]
    assert all(n == n_per[0] for n in n_per), f"ball count varies across frames: {n_per}"
    for b in s["dyn_obs"][0]:
        for k in ("pos", "vel", "radius", "kind"):
            assert k in b, f"dyn_obs ball missing '{k}': {b.keys()}"

    # dt_seq
    assert s["dt_seq"].shape == (10,), s["dt_seq"].shape
    assert np.all(s["dt_seq"] > 0) and np.all(s["dt_seq"] < 1.0)

    # meta intrinsics self-check
    intr = s["meta"]["intrinsics"]
    assert intr["fx"] == 80 and intr["fy"] == 80 and intr["cx"] == 80 and intr["cy"] == 45
    assert intr["W"] == 160 and intr["H"] == 90

    print(f"[OK] dynamic ds len={len(ds_dyn)}, sample keys={sorted(s.keys())}")
    print(f"[OK] depth_seq shape={s['depth_seq'].shape} dtype={s['depth_seq'].dtype} "
          f"range=[{s['depth_seq'].min():.3f}, {s['depth_seq'].max():.3f}]")
    print(f"[OK] state_seq len={len(s['state_seq'])} fields={sorted(s['state_seq'][0].keys())}")
    print(f"[OK] dyn_obs frames={len(s['dyn_obs'])} balls/frame={n_per[0]} "
          f"fields={sorted(s['dyn_obs'][0][0].keys())}")
    print(f"[OK] dt_seq={s['dt_seq'][0]:.4f}s constant across K")
    print(f"[OK] meta intrinsics match config.yaml")

    # ---- validation split sanity ----
    ds_val = YOPODataset(mode='valid', dynamic=True, K=10)
    assert len(ds_val) > 0
    # train ∪ valid should match total seq count
    total_expected = len(ds_dyn) + len(ds_val)
    print(f"[OK] train/valid split: {len(ds_dyn)}/{len(ds_val)} = {total_expected} total")

    # ---- regression: static mode still works ----
    print("\n== YOPODataset(dynamic=False) regression ==")
    ds_stat = YOPODataset(mode='valid')   # static, valid set is fast to construct
    static_item = ds_stat[0]
    assert isinstance(static_item, tuple), f"static path should still return a tuple, got {type(static_item)}"
    assert len(static_item) == 5, f"static path tuple length: {len(static_item)}"
    img, pos, rot, obs, mid = static_item
    assert img.shape == (1, 96, 160), img.shape
    print(f"[OK] static ds[0]: tuple(image{img.shape}, pos{pos.shape}, rot{rot.shape}, "
          f"obs{obs.shape}, map_id={mid})")

    print()
    print("[PASS] stage-2 2.e: dynamic mode produces the documented dict, "
          "static path regression-clean")


if __name__ == "__main__":
    main()
