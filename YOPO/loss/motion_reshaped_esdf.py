"""REACT stage-3.1 [🟧] motion-reshaped distance field loss.

Penalizes proximity to dynamic obstacles, with the *signed-relative-velocity*
projection along the obstacle->trajectory direction folded into a re-shaping
of the geometric distance. Approaching obstacles are penalized more strongly
than receding ones at the same Euclidean distance.

Reference: REACT 实施指南 §3.4 (motivated by Flow-Aided distance fields).

Math
----
For each waypoint `p` on the predicted trajectory and each obstacle
`(p_obs, v_obs, r)`:

    rel_v       = v_self - v_obs                     # m/s, world frame
    dir_to_obs  = normalize(p_obs - p)               # unit vector traj -> obs
    closing     = relu( rel_v · dir_to_obs )         # >= 0; signed speed at
                                                     # which the gap to the
                                                     # obstacle is shrinking
    d_geo       = || p - p_obs || - r                # signed metric distance
                                                     # to the obstacle surface
    d_reshape   = d_geo / (1 + alpha * closing)      # smaller when closing in

    penalty     = softplus(d_safe - d_reshape)       # smooth hinge

Total loss = mean over (waypoints, real obstacles).

Note on the sign convention: REACT 实施指南 §3.4 pseudocode used
`dir_to_self = (traj - p_obs)`, which gives `rel_v · dir_to_self > 0` when
the gap is *growing* (receding), not closing.  The intent of the
docstring there ("朝向自己运动的障碍惩罚加倍" / "obstacles moving toward us
get doubled penalty") matches the corrected sign used here.  Derivation:
d/dt(||obs - traj||) = dir(traj->obs) · (v_obs - v_self), so the gap
*shrinks* when dir_to_obs · (v_self - v_obs) > 0.

Receding obstacles (closing <= 0) get reshape factor 1, so the loss
collapses to softplus(d_safe - d_geo) -- equivalent to a plain ESDF
collision loss at the same geometric distance.  Closing-in obstacles get
a *smaller* d_reshape, hence a *larger* penalty, even at the same
current geometric distance.

API
---
trajectory : (B, N, 3) float        N waypoints per batch element
v_self     : (B, 3) float           drone world-frame velocity per batch element
obstacles  : (B, M, 7) float        [px, py, pz, vx, vy, vz, r] per obstacle
obs_mask   : (B, M)    bool         True where the obstacle slot is real
                                    (padding-aware; M_max obstacle slots)

Returns
-------
loss : scalar
"""
import torch
import torch.nn.functional as F


def motion_reshaped_collision_loss(
    trajectory: torch.Tensor,   # (B, N, 3) waypoint positions in world frame
    v_self:     torch.Tensor,   # (B, 3)    drone world-frame velocity
    obstacles:  torch.Tensor,   # (B, M, 7) packed dyn-obs: pos(3), vel(3), r(1)
    obs_mask:   torch.Tensor,   # (B, M)    True for real obstacle, False padding
    alpha:      float = 2.0,
    d_safe:     float = 0.6,
    eps:        float = 1e-6,
) -> torch.Tensor:
    """Returns the mean motion-reshaped collision penalty over (batch,
    waypoints, real obstacles). Returns a zero scalar (with grad) when
    obs_mask has no True values, so static-only batches contribute 0."""
    B, N, _ = trajectory.shape
    M = obstacles.shape[1]

    p_obs = obstacles[..., 0:3]                                 # (B, M, 3)
    v_obs = obstacles[..., 3:6]                                 # (B, M, 3)
    r     = obstacles[..., 6:7]                                 # (B, M, 1)

    # delta = obs - traj (traj -> obs vector): see Math section for the sign
    # rationale.  shape: (B, 1, M, 3) - (B, N, 1, 3) -> (B, N, M, 3)
    delta = p_obs.unsqueeze(1) - trajectory.unsqueeze(2)
    dist = delta.norm(dim=-1, keepdim=True).clamp(min=eps)      # (B, N, M, 1)
    dir_to_obs = delta / dist                                    # (B, N, M, 3)
    d_geo = (dist.squeeze(-1) - r.squeeze(-1).unsqueeze(1))      # (B, N, M)

    # rel_v is per-batch, per-obstacle (not per-waypoint).  Broadcast to (B, N, M, 3).
    rel_v = (v_self.unsqueeze(1) - v_obs).unsqueeze(1)           # (B, 1, M, 3)
    closing = (rel_v * dir_to_obs).sum(dim=-1)                   # (B, N, M)
    closing = closing.clamp(min=0.0)                             # only shrinking gap

    d_reshape = d_geo / (1.0 + alpha * closing)                  # (B, N, M)
    penalty = F.softplus(d_safe - d_reshape)                     # (B, N, M)

    # zero out padding rows; (B, 1, M) broadcast across N
    mask = obs_mask.unsqueeze(1).to(penalty.dtype)               # (B, 1, M)
    penalty = penalty * mask

    n_real = mask.sum().clamp(min=1.0) * N
    return penalty.sum() / n_real
