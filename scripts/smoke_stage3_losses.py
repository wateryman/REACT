"""Stage-3.1 smoke test for the two new dynamic-loss modules.

Each test exercises a symbol/sign property that must hold for the loss to
be useful as a training signal, *and* that gradients flow back through
the input.  Failures here mean the loss math is wrong, not that training
won't converge -- the trainer integration test in commit C catches that.

Run from REACT/ root:
    python scripts/smoke_stage3_losses.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YOPO_DIR = os.path.join(HERE, "..", "YOPO")
os.chdir(YOPO_DIR)
sys.path.insert(0, YOPO_DIR)

import torch

from loss.motion_reshaped_esdf import motion_reshaped_collision_loss
from loss.kinodynamic_loss import kinodynamic_loss


# ============================================================================
# motion_reshaped_collision_loss
# ============================================================================

def test_motion_reshaped_approach_greater_than_recede():
    """Same drone, same trajectory, same obstacle starting position+radius,
    same |v_obs|.  Two scenarios: obstacle moves toward the trajectory
    (approach) vs away from it (recede).  Approach must be strictly more
    penalized.

    Setup: trajectory is at x=2..6 on the +X axis (all on the -X side of
    the obstacle).  Obstacle starts at x=10.  v_obs is +-5 along X.
    "Approach" therefore means obstacle moves -X (toward the trajectory),
    "recede" means +X.  This keeps the sign of the closing speed
    consistent across all waypoints.
    """
    B, N, M = 1, 5, 1
    traj = torch.tensor([[[2.0, 0, 1], [3.0, 0, 1], [4.0, 0, 1], [5.0, 0, 1], [6.0, 0, 1]]])
    v_self = torch.tensor([[3.0, 0, 0]])
    obs_approach = torch.tensor([[[10.0, 0, 1, -5.0, 0, 0, 0.5]]])  # obstacle moves -X toward traj
    obs_recede   = torch.tensor([[[10.0, 0, 1, +5.0, 0, 0, 0.5]]])  # obstacle moves +X away from traj
    mask = torch.ones(B, M, dtype=torch.bool)
    L_app = motion_reshaped_collision_loss(traj, v_self, obs_approach, mask)
    L_rec = motion_reshaped_collision_loss(traj, v_self, obs_recede, mask)
    print(f"  approach loss = {L_app.item():.4f}, recede loss = {L_rec.item():.4f}")
    assert L_app > L_rec * 1.05, f"approach {L_app:.4f} must exceed recede {L_rec:.4f} by >5%"
    return L_app, L_rec


def test_motion_reshaped_gradient_flows():
    """Gradient must flow back into the trajectory (otherwise the loss
    has no effect on training)."""
    traj = torch.tensor([[[2.0, 0, 1], [3.0, 0, 1], [4.0, 0, 1]]], requires_grad=True)
    v_self = torch.tensor([[3.0, 0, 0]])
    obs = torch.tensor([[[4.0, 0, 1, -5.0, 0, 0, 0.5]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    L = motion_reshaped_collision_loss(traj, v_self, obs, mask)
    L.backward()
    grad_max = float(traj.grad.abs().max())
    print(f"  d(loss)/d(traj) max-abs = {grad_max:.4e}")
    assert grad_max > 1e-6, "trajectory grad is zero -- loss is decoupled"
    return grad_max


def test_motion_reshaped_padding_is_skipped():
    """Padded (mask=False) obstacle slots must not contribute. Compare
    a batch with the padded slot zeroed-out vs filled with junk."""
    traj = torch.tensor([[[2.0, 0, 1], [3.0, 0, 1], [4.0, 0, 1]]])
    v_self = torch.tensor([[3.0, 0, 0]])
    obs_real = torch.tensor([[[4.0, 0, 1, -5.0, 0, 0, 0.5]]])
    obs_padded = torch.tensor([[[4.0, 0, 1, -5.0, 0, 0, 0.5],
                                 [3.5, 0, 1, +5.0, 0, 0, 0.4]]])
    mask_real = torch.ones(1, 1, dtype=torch.bool)
    mask_padded = torch.tensor([[True, False]])
    L1 = motion_reshaped_collision_loss(traj, v_self, obs_real, mask_real)
    L2 = motion_reshaped_collision_loss(traj, v_self, obs_padded, mask_padded)
    print(f"  L(real)={L1.item():.6f}  L(real + masked padding)={L2.item():.6f}")
    assert abs(L1 - L2) < 1e-5, "padded slot leaked into loss"


def test_motion_reshaped_empty_batch():
    """All-padding batch (zero real obstacles) must produce a finite zero
    loss with no nan/inf, useful when mixing static frames into training."""
    traj = torch.zeros(2, 3, 3, requires_grad=True)
    v_self = torch.zeros(2, 3)
    obs = torch.zeros(2, 4, 7)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    L = motion_reshaped_collision_loss(traj, v_self, obs, mask)
    assert torch.isfinite(L), f"loss not finite: {L}"
    print(f"  empty-batch loss = {L.item():.6f} (finite, ok for static frames)")


# ============================================================================
# kinodynamic_loss
# ============================================================================

def test_kinodynamic_within_bounds_zero():
    """Trajectory inside the v/a/j envelope -> total loss is exactly 0."""
    wp = torch.zeros(1, 5, 9)
    wp[:, :, 3] = 5.0    # vx = 5  < v_max 8
    wp[:, :, 6] = 5.0    # ax = 5  < a_max 10
    dt = torch.full((1, 5), 0.1)
    total, comp = kinodynamic_loss(wp, dt, v_max=8.0, a_max=10.0, j_max=30.0)
    print(f"  in-bounds total={total.item():.6f}  components={ {k: v.item() for k, v in comp.items()} }")
    assert total.item() == 0.0, f"expected 0 inside envelope, got {total}"


def test_kinodynamic_v_violation_positive():
    """v_max violation alone produces a positive vel component."""
    wp = torch.zeros(1, 5, 9)
    wp[:, :, 3] = 12.0   # vx = 12 > v_max 8 -> violation 4
    dt = torch.full((1, 5), 0.1)
    total, comp = kinodynamic_loss(wp, dt, v_max=8.0, a_max=10.0, j_max=30.0)
    print(f"  v-violation total={total.item():.4f}  vel_pen={comp['vel'].item():.4f}")
    assert comp["vel"].item() == 4.0, f"expected vel pen=4, got {comp['vel']}"
    assert total.item() > 0


def test_kinodynamic_gradient_flows():
    """Gradient flows from the loss into vel/acc/jerk-violating waypoints."""
    wp = torch.zeros(1, 5, 9, requires_grad=True)
    with torch.no_grad():
        wp[:, :, 3] = 15.0       # large vx -> violation
    dt = torch.full((1, 5), 0.1)
    total, _ = kinodynamic_loss(wp, dt, v_max=8.0, a_max=10.0, j_max=30.0)
    total.backward()
    grad_max = float(wp.grad.abs().max())
    print(f"  d(loss)/d(wp) max-abs = {grad_max:.4e}")
    assert grad_max > 1e-6, "waypoints grad is zero"


# ============================================================================

if __name__ == "__main__":
    print("== motion_reshaped_collision_loss ==")
    print(" T1 approach > recede"); test_motion_reshaped_approach_greater_than_recede()
    print(" T2 gradient flows");    test_motion_reshaped_gradient_flows()
    print(" T3 padding mask");      test_motion_reshaped_padding_is_skipped()
    print(" T4 empty batch");       test_motion_reshaped_empty_batch()
    print()
    print("== kinodynamic_loss ==")
    print(" T5 within bounds = 0"); test_kinodynamic_within_bounds_zero()
    print(" T6 v violation > 0");   test_kinodynamic_v_violation_positive()
    print(" T7 gradient flows");    test_kinodynamic_gradient_flows()
    print()
    print("[PASS] stage-3.1 loss modules: 7/7 checks")
