"""REACT stage-3.1 [🟧] kinodynamic-consistency loss.

Penalizes trajectories that exceed the drone's velocity / acceleration /
jerk envelope. The envelope values come from the controller's tracking
capability; staying inside it is a prerequisite for the controller to
actually realize the planned trajectory.

Reference: REACT 实施指南 §3.5 (CLDM-inspired).

Inputs
------
waypoints   : (B, N, 9)  per-waypoint [px,py,pz, vx,vy,vz, ax,ay,az]
                          world frame, returned by the YopoHead -> primitive
                          mapping (already in metric units).
time_intervals : (B, N)  seconds elapsed at each waypoint (cumulative or
                          per-segment; here we use per-segment dt_k between
                          waypoint k and k+1).

Caps
----
v_max, a_max, j_max : scalars (m/s, m/s^2, m/s^3 respectively).

Behaviour
---------
- velocity penalty: ReLU(|vel_k| - v_max), mean over (B, N)
- accel penalty:    ReLU(|acc_k| - a_max), mean over (B, N)
- jerk penalty:     finite-difference jerk = (acc_{k+1} - acc_k) / dt_k;
                    ReLU(|jerk| - j_max), mean over (B, N-1)
- Final loss = sum of the three means.

The three components are returned alongside the total so the trainer can
log them separately if it wants.
"""
import torch
import torch.nn.functional as F


def kinodynamic_loss(
    waypoints:      torch.Tensor,   # (B, N, 9)
    time_intervals: torch.Tensor,   # (B, N)
    v_max:          float = 8.0,
    a_max:          float = 10.0,
    j_max:          float = 30.0,
    eps:            float = 1e-3,
):
    """Returns (total, components_dict). The total is a single scalar
    suitable for backward(); components are detached scalars for logging."""
    vel = waypoints[..., 3:6]                                   # (B, N, 3)
    acc = waypoints[..., 6:9]                                   # (B, N, 3)
    dt = time_intervals.clamp(min=eps).unsqueeze(-1)             # (B, N, 1)

    vel_norm = vel.norm(dim=-1)                                  # (B, N)
    acc_norm = acc.norm(dim=-1)                                  # (B, N)

    vel_pen = F.relu(vel_norm - v_max).mean()
    acc_pen = F.relu(acc_norm - a_max).mean()

    # Finite-difference jerk between consecutive waypoints
    if waypoints.shape[1] >= 2:
        jerk = (acc[:, 1:] - acc[:, :-1]) / dt[:, :-1]           # (B, N-1, 3)
        jerk_pen = F.relu(jerk.norm(dim=-1) - j_max).mean()
    else:
        jerk_pen = torch.zeros((), device=waypoints.device, dtype=waypoints.dtype)

    total = vel_pen + acc_pen + jerk_pen
    return total, dict(vel=vel_pen.detach(),
                       acc=acc_pen.detach(),
                       jerk=jerk_pen.detach())
