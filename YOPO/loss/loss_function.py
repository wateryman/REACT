import math
import torch as th
import torch.nn as nn
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss
from loss.motion_reshaped_esdf import motion_reshaped_collision_loss
from loss.kinodynamic_loss import kinodynamic_loss as _kinodynamic_loss_fn


class YOPOLoss(nn.Module):
    def __init__(self):
        """
        Compute the cost: including smoothness, safety, guidance, goal cost, etc.
        Currently, keeping multi-segment polynomial support (not yet verified), but only using a single-segment polynomial (m = 1) for now.
        dp: decision parameters
        df: fixed parameters
        """
        super(YOPOLoss, self).__init__()
        self.sgm_time = cfg["sgm_time"]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self._C, self._B, self._L, self._RJ, self._RA = self.qp_generation()
        self._RJ = self._RJ.to(self.device)
        self._RA = self._RA.to(self.device)
        self._L = self._L.to(self.device)
        self.denormalize_weight()
        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        self.safety_loss = SafetyLoss(self._L)
        self.goal_loss = GuidanceLoss()
        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print("-------------------------")

    def qp_generation(self):
        # 论文中的映射矩阵
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        # H海森矩阵，对应Jerk
        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        # Q海森矩阵，对应Accel
        Q = th.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (self.sgm_time ** (i + j - 3))

        return self.stack_opt_dep(A, H, Q)

    def stack_opt_dep(self, A, H, Q):
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        _C = th.transpose(Ct, 0, 1)

        B = th.inverse(A)

        B_T = th.transpose(B, 0, 1)

        _L = B @ Ct

        _R_Jerk = _C @ (B_T) @ H @ B @ Ct

        _R_Acc = _C @ (B_T) @ Q @ B @ Ct

        return _C, B, _L, _R_Jerk, _R_Acc

    def denormalize_weight(self):
        """
        Denormalize the cost weight to ensure consistency across different speeds to simplify parameter tuning.
        smoothness cost: time integral of jerk² is used as a smoothness cost.
                         If the speed is scaled by n, the cost is scaled by n⁵ (because jerk * n⁶ and time * 1/n).
        safety cost:     time integral of the distance from trajectory to obstacles.
                         If the speed is scaled by n, the cost is scaled by 1/n (because time * 1/n).
        goal cost:       projection of the trajectory onto goal direction.
                         Independent of speed.
        """
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"] / vel_scale ** 5
        self.accele_weight = cfg["wa"] / vel_scale ** 3
        self.safety_weight = cfg["wc"]
        self.goal_weight = cfg["wg"]

    def forward(self, state, prediction, goal, map_id):
        """
        Args:
            prediction: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            state: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            map_id: (batch_size) which ESDF map to query

        Returns:
            cost: (batch_size) → weighted cost
        """
        # Fixed part: initial pos, vel, acc → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Df = state.permute(0, 2, 1)

        # Decision parameters (local frame) → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Dp = prediction.permute(0, 2, 1)

        smoothness_cost, acceleration_cost = self.smoothness_loss(Df, Dp)
        safety_cost = self.safety_loss(Df, Dp, map_id)
        goal_cost = self.goal_loss(Df, Dp, goal)

        return self.smoothness_weight * smoothness_cost, self.safety_weight * safety_cost, self.goal_weight * goal_cost, self.accele_weight * acceleration_cost

    @staticmethod
    def z_floor_loss(end_pos_w, z_floor: float = 0.3, lam_floor: float = 1.0):
        """🟦 stage-5.B plan B: penalise endstate world-frame z below a floor.

        The motivating observation: the SafetyLoss queries an ESDF computed
        from a static point cloud that has no ground plane.  For any
        endstate prediction with z < 0 (below the lowest cloud point), the
        ESDF returns a large free-space distance -> low safety cost ->
        the score head learns to prefer "look down" anchors.  See
        REACT_MATH_Derivations/01_collision_loss_saturation.tex §4 and
        the stage-5.B closed-loop debug log for the empirical signature
        (C1 trained model predicts end_z ~ -2 m and dives below ground).

        Loss: quadratic soft hinge below z_floor, zero above.  The
        gradient pushes the predicted endstate UP when it dips below
        z_floor.

        Args
        ----
        end_pos_w : (B*V*H, 3) world-frame predicted endstate position
        z_floor   : minimum allowed world-frame z (m).  Default 0.3 m
                    matches the bake's ball-z floor (bbox_z_lo).
        lam_floor : multiplier applied to the squared hinge.

        Returns
        -------
        Already-weighted scalar (so the caller does
        `loss_total += yopo_loss.z_floor_loss(...)`).
        """
        return lam_floor * th.relu(z_floor - end_pos_w[..., 2]).pow(2).mean()

    @staticmethod
    def revae_loss(recon, target, mu, logvar, lam_recon: float = 1.0, lam_kl: float = 1e-3):
        """🟩 PEMTRS reVAE loss = lam_recon * MSE(recon, target) + lam_kl * KL.

        Sobel-edge component is added in stage 3 (losses/sobel_loss.py).
        """
        mse = th.nn.functional.mse_loss(recon, target)
        # standard normal prior KL, mean-reduced over batch and latent
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        return lam_recon * mse + lam_kl * kl

    @staticmethod
    def dyn_collision_loss(trajectory, v_self, obstacles, obs_mask,
                            lam_dyn: float = 1.0, alpha: float = 2.0, d_safe: float = 0.6):
        """🟧 stage-3.1: weighted motion-reshaped collision loss.

        Mirrors revae_loss's lam-in-kwarg pattern: the function returns the
        already-weighted scalar so callers can sum without double-bookkeeping.
        Pure math is in loss/motion_reshaped_esdf.py.

        When obs_mask has zero True entries (i.e. a static-only batch with
        no dynamic obstacles), the underlying loss is 0; this method
        therefore composes safely into a single training step that mixes
        static and dynamic samples.
        """
        return lam_dyn * motion_reshaped_collision_loss(
            trajectory, v_self, obstacles, obs_mask, alpha=alpha, d_safe=d_safe)

    @staticmethod
    def kinodynamic_loss(waypoints, time_intervals,
                          lam_kino: float = 1.0,
                          v_max: float = 8.0, a_max: float = 10.0, j_max: float = 30.0):
        """🟧 stage-3.1: weighted kinodynamic envelope loss + per-component dict.

        Returns (lam_kino * total, components_dict).  The components are
        detached scalars (vel/acc/jerk) so the trainer can log each on
        tensorboard without affecting backprop.
        Pure math is in loss/kinodynamic_loss.py.
        """
        total, comp = _kinodynamic_loss_fn(
            waypoints, time_intervals, v_max=v_max, a_max=a_max, j_max=j_max)
        return lam_kino * total, comp