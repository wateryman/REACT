"""
Training Strategy
supervised learning, imitation learning, testing, rollout
"""
import os
import time
import atexit
from torch.nn import functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import YOPOLoss
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import *


class YopoTrainer:
    def __init__(
            self,
            learning_rate=0.001,
            batch_size=32,
            loss_weight=[],
            tensorboard_path=None,
            checkpoint_path=None,
            save_on_exit=False,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_weight = loss_weight
        if save_on_exit: self._exit_func = atexit.register(self.save_model)
        # logger
        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        # params
        self.traj_num = cfg['traj_num']

        # network
        print("Loading network...")
        # 🟧 stage-3.2 sub-C: optional DCA side channel.  When cfg flag is off,
        # YopoNetwork(use_dca=False) matches stage-3.1 byte-for-byte.
        use_dca = bool(cfg["dynamic_attention"]["enable"])
        dca_n_heads = int(cfg["dynamic_attention"]["n_heads"])
        # 🟧 stage-4: read revae.enable so the ablation can disable the
        # auxiliary encoder for the original-YOPO baseline row.
        use_revae = bool(cfg["revae"]["enable"])
        self.policy = YopoNetwork(use_revae=use_revae,
                                   revae_latent=int(cfg["revae"]["latent_dim"]),
                                   use_dca=use_dca, dca_n_heads=dca_n_heads)
        if use_dca:
            print(f"DCA side channel: ENABLED (n_heads={dca_n_heads})")
        if not use_revae:
            print(f"reVAE: DISABLED (ablation mode)")
        self.policy = self.policy.to(self.device)
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self.policy.load_state_dict(state_dict)
            print("Checkpoint ", checkpoint_path, " loaded successfully")
        except FileNotFoundError:
            print("Training from scratch")

        # loss
        self.yopo_loss = YOPOLoss()

        # optimizer
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=True)
        print("Network Loaded! Loading Dataset...")

        # dataset (you can adjust num_workers according to your training speed)
        self.train_dataloader = DataLoader(YOPODataset(mode='train'), batch_size=self.batch_size, shuffle=True,
                                           num_workers=4, pin_memory=True)
        self.val_dataloader = DataLoader(YOPODataset(mode='valid'), batch_size=self.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)

        # 🟧 stage-3.1 C.3: optional dynamic dataloader.  dynamic_ratio == 0 -->
        # static-only training (regression-clean: dyn_obs is never passed and
        # forward_and_compute_loss returns dyn = kino = 0 exactly, matching C.2 smoke).
        self.dynamic_ratio = float(cfg["dynamic_ratio"])
        if self.dynamic_ratio > 0.0:
            from policy.yopo_dataset import DynamicYOPOWrapper
            self.dyn_train_dataloader = DataLoader(
                DynamicYOPOWrapper(mode='train'),
                batch_size=self.batch_size, shuffle=True,
                num_workers=2, pin_memory=True)
            self.dyn_train_iter = iter(self.dyn_train_dataloader)
            print(f"Dynamic dataloader enabled (ratio={self.dynamic_ratio:.2f}, "
                  f"{len(self.dyn_train_dataloader.dataset)} seqs).")
        else:
            self.dyn_train_dataloader = None
            self.dyn_train_iter = None
        print("Dataset Loaded!")

    def train(self, epoch, save_interval=None):
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.progress_log.console.log("Saving model...")
                    policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
                    torch.save(self.policy.state_dict(), policy_path)
            self.progress_log.console.log("Train YOPO Finish!")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch: int, total_progress):
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=len(self.train_dataloader))
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        traj_losses, score_losses, revae_losses, dyn_losses, kino_losses, smooth_losses, safety_losses, goal_losses, acc_losses, start_time = [], [], [], [], [], [], [], [], [], time.time()
        lam_vae = cfg["loss_weights"]["lam_vae"]
        for step, static_batch in enumerate(self.train_dataloader):  # obs: body frame
            if static_batch[0].shape[0] != self.batch_size:  continue  # batch size == number of env

            # 🟧 stage-3.1 C.3: per-step swap to a dynamic batch with probability
            # dynamic_ratio.  Static is always the "main loop" so that the total
            # step count == number of static batches; we never train on more
            # epochs of the (smaller) dynamic dataset than this implies.
            dyn_obs_payload = None
            if self.dyn_train_iter is not None and torch.rand(1).item() < self.dynamic_ratio:
                try:
                    dyn_batch = next(self.dyn_train_iter)
                except StopIteration:
                    self.dyn_train_iter = iter(self.dyn_train_dataloader)
                    dyn_batch = next(self.dyn_train_iter)
                if dyn_batch[0].shape[0] == self.batch_size:
                    depth, pos, rot, obs_b, map_id, dyn_pad, dyn_mask = dyn_batch
                    dyn_obs_payload = (dyn_pad, dyn_mask)
                else:
                    depth, pos, rot, obs_b, map_id = static_batch
            else:
                depth, pos, rot, obs_b, map_id = static_batch

            self.optimizer.zero_grad()

            # 🟧 stage-3.1: 9-tuple includes dyn_loss + kino_loss.  Both are 0 when
            # dyn_obs_payload is None (static batch); positive when dynamic.
            (trajectory_loss, score_loss, revae_loss, dyn_loss, kino_loss,
             smooth_cost, safety_cost, goal_cost, acc_cost) = self.forward_and_compute_loss(
                depth, pos, rot, obs_b, map_id, dyn_obs=dyn_obs_payload)

            # dyn_loss and kino_loss are already lam-weighted by YOPOLoss wrappers.
            loss = (self.loss_weight[0] * trajectory_loss
                    + self.loss_weight[1] * score_loss
                    + lam_vae * revae_loss
                    + dyn_loss
                    + kino_loss)

            # Optimize the policy
            loss.backward()
            self.optimizer.step()

            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            score_losses.append(self.loss_weight[1] * score_loss.item())
            revae_losses.append(lam_vae * revae_loss.item())
            dyn_losses.append(dyn_loss.item())
            kino_losses.append(kino_loss.item())
            smooth_losses.append(self.loss_weight[0] * smooth_cost.item())
            safety_losses.append(self.loss_weight[0] * safety_cost.item())
            goal_losses.append(self.loss_weight[0] * goal_cost.item())
            acc_losses.append(self.loss_weight[0] * acc_cost.item())

            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                self.progress_log.console.log(f"Epoch: {epoch}, Traj Loss: {np.mean(traj_losses):.3g}, "
                                              f"Score Loss: {np.mean(score_losses):.3g}, "
                                              f"reVAE Loss: {np.mean(revae_losses):.3g}, "
                                              f"Dyn Loss: {np.mean(dyn_losses):.3g}, "
                                              f"Kino Loss: {np.mean(kino_losses):.3g} "
                                              f"Batch FPS: {batch_fps:.3g}")
                global_step = epoch * len(self.train_dataloader) + step
                self.tensorboard_log.add_scalar("Train/TrajLoss",  np.mean(traj_losses),  global_step)
                self.tensorboard_log.add_scalar("Train/ScoreLoss", np.mean(score_losses), global_step)
                self.tensorboard_log.add_scalar("Train/ReVAELoss", np.mean(revae_losses), global_step)
                self.tensorboard_log.add_scalar("Train/DynLoss",   np.mean(dyn_losses),   global_step)
                self.tensorboard_log.add_scalar("Train/KinoLoss",  np.mean(kino_losses),  global_step)
                self.tensorboard_log.add_scalar("Detail/SmoothLoss", np.mean(smooth_losses), global_step)
                self.tensorboard_log.add_scalar("Detail/SafetyLoss", np.mean(safety_losses), global_step)
                self.tensorboard_log.add_scalar("Detail/GoalLoss",   np.mean(goal_losses),   global_step)
                self.tensorboard_log.add_scalar("Detail/AccelLoss",  np.mean(acc_losses),    global_step)
                traj_losses, score_losses, revae_losses, dyn_losses, kino_losses, smooth_losses, safety_losses, goal_losses, acc_losses, start_time = [], [], [], [], [], [], [], [], [], time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        self.progress_log.remove_task(one_epoch_progress)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        traj_losses, score_losses = [], []
        for step, (depth, pos, rot, obs_b, map_id) in enumerate(self.val_dataloader):  # obs: body frame
            if depth.shape[0] != self.batch_size:  continue  # batch size == num of env

            (trajectory_loss, score_loss, _, _, _, _, _, _, _) = self.forward_and_compute_loss(
                depth, pos, rot, obs_b, map_id)

            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            score_losses.append(self.loss_weight[1] * score_loss.item())
            self.progress_log.update(one_epoch_progress, advance=1)

        self.progress_log.console.log(f"Eval: {epoch}, Traj Loss: {np.mean(traj_losses):.3g}, Score Loss: {np.mean(score_losses):.3g} ")
        self.tensorboard_log.add_scalar("Eval/TrajLoss", np.mean(traj_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/ScoreLoss", np.mean(score_losses), epoch)
        self.progress_log.remove_task(one_epoch_progress)

    def forward_and_compute_loss(self, depth, pos, rot, obs_b, map_id, dyn_obs=None):
        """🟧 stage-3.1 C.2: forward + losses with optional dynamic-obstacle path.

        dyn_obs : tuple (obstacles_tensor, obs_mask_tensor) or None
            obstacles_tensor : (B, M, 7) -- packed [px,py,pz, vx,vy,vz, radius]
            obs_mask_tensor  : (B, M) bool -- True for real slots
            When None (default), both dyn_collision_loss and kinodynamic_loss
            return zero, so static-only training stays bit-identical to the
            pre-C.2 trainer.  C.3 wires a dynamic dataloader that fills this.

        Returns
        -------
        9-tuple (traj, score, revae, dyn, kino, smooth, safety, goal, acc)
        where dyn and kino are already lam-weighted by the YOPOLoss wrappers.
        """
        depth, pos, rot, obs_b, map_id = [x.to(self.device) for x in [depth, pos, rot, obs_b, map_id]]

        # 1. pre-process
        goal_w, start_vel_w, start_acc_w = state_body2world(pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        # 🟧 stage-3.2 sub-C: when a dynamic batch is supplied, build per-anchor
        # cross-attention tokens (drone-relative position + absolute world vel +
        # radius) and pass to the policy.  The wrapper has shape (B, M, 7) world.
        # We subtract the drone world position to get relative position; vel and
        # radius pass through unchanged.  No rotation -- network learns yaw
        # invariance from its random-obs training distribution.
        dyn_tokens_for_net = None
        dyn_mask_for_net = None
        if dyn_obs is not None:
            obstacles_w, obs_mask = dyn_obs
            obstacles_w = obstacles_w.to(self.device)
            obs_mask = obs_mask.to(self.device)
            rel_pos = obstacles_w[..., 0:3] - pos.unsqueeze(1)      # (B, M, 3)
            abs_vel = obstacles_w[..., 3:6]                          # (B, M, 3)
            radius  = obstacles_w[..., 6:7]                          # (B, M, 1)
            dyn_tokens_for_net = torch.cat([rel_pos, abs_vel, radius], dim=-1)
            dyn_mask_for_net = obs_mask

        # 2. forward propagation (REACT: inference now returns reVAE recon/mu/logvar too)
        endstate, score, recon, mu, logvar = self.policy.inference(
            depth, obs_b,
            dyn_obs_tokens=dyn_tokens_for_net,
            dyn_obs_mask=dyn_mask_for_net)

        # 3. post-process [B, V, H, 9] -> [B*V*H, 9]
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(self.batch_size * self.traj_num, 9)
        score_flat = score.reshape(self.batch_size * self.traj_num)

        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        start_state_w_exp = start_state_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        goal_w = goal_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]

        # [B*V*H, 3] [B*V*H, 3] [B*V*H, 3]
        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded, rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9]
        )
        # [B*V*H, 3, 3]: [px, py, pz; vx, vy, vz; ax, ay, az]
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        smooth_cost, safety_cost, goal_cost, acc_cost = self.yopo_loss(start_state_w_exp, end_state_w, goal_w, map_id)
        trajectory_loss = (smooth_cost + safety_cost + goal_cost + acc_cost).mean()

        score_label = (smooth_cost + safety_cost + goal_cost + acc_cost).clone().detach()
        score_loss = F.smooth_l1_loss(score_flat, score_label)

        # 🟩 reVAE loss (skipped when use_revae=False)
        if recon is not None:
            lam_recon = cfg["revae"]["lam_recon"]
            lam_kl = cfg["revae"]["lam_kl"]
            revae_loss = self.yopo_loss.revae_loss(recon, depth, mu, logvar,
                                                    lam_recon=lam_recon, lam_kl=lam_kl)
        else:
            revae_loss = torch.zeros((), device=self.device)

        # 🟧 stage-3.1: dyn-obs-aware losses.  Both gated on dyn_obs presence
        # so static-only batches contribute exactly zero (regression-clean
        # versus pre-C.2 baseline).  C.3 fills dyn_obs from the dynamic
        # dataloader; until then this branch is never taken.
        if dyn_obs is not None:
            obstacles, obs_mask = dyn_obs
            obstacles = obstacles.to(self.device)
            obs_mask = obs_mask.to(self.device)
            # Expand to per-anchor: (B, M, 7) -> (B*V*H, M, 7); same for mask
            obs_expanded = obstacles.repeat_interleave(self.traj_num, dim=0)
            mask_expanded = obs_mask.repeat_interleave(self.traj_num, dim=0)
            # One waypoint per anchor (the predicted endstate position)
            trajectory_pts = end_pos_w.unsqueeze(1)              # (B*V*H, 1, 3)
            # v_self for closing-speed is the drone's CURRENT velocity (NOT
            # the predicted endstate velocity), expanded per anchor.
            v_self_exp = start_vel_w.repeat_interleave(self.traj_num, dim=0)  # (B*V*H, 3)
            dyn_loss = self.yopo_loss.dyn_collision_loss(
                trajectory_pts, v_self_exp, obs_expanded, mask_expanded,
                lam_dyn=cfg["loss_weights"]["lam_dyn"],
                alpha=cfg["motion_reshaped"]["alpha"],
                d_safe=cfg["motion_reshaped"]["d_safe"])
            # Kinodynamic envelope on the endstate (N=1 -> jerk skipped).
            # end_state_w is (B*V*H, 3, 3) in [pos_row, vel_row, acc_row] x [x,y,z].
            # Flatten to (B*V*H, 9): pos(3) | vel(3) | acc(3).
            wp_for_kino = end_state_w.reshape(end_state_w.shape[0], 9).unsqueeze(1)  # (B*V*H, 1, 9)
            dt_dummy = torch.ones(wp_for_kino.shape[0], 1, device=self.device) * 0.1  # ignored when N=1
            kino_loss, _ = self.yopo_loss.kinodynamic_loss(
                wp_for_kino, dt_dummy,
                lam_kino=cfg["loss_weights"]["lam_kino"],
                v_max=cfg["kinodynamic"]["v_max"],
                a_max=cfg["kinodynamic"]["a_max"],
                j_max=cfg["kinodynamic"]["j_max"])
        else:
            dyn_loss = torch.zeros((), device=self.device)
            kino_loss = torch.zeros((), device=self.device)

        return (trajectory_loss, score_loss, revae_loss, dyn_loss, kino_loss,
                smooth_cost.mean(), safety_cost.mean(), goal_cost.mean(), acc_cost.mean())

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
            torch.save(self.policy.state_dict(), policy_path)
            atexit.unregister(self._exit_func)

    def get_next_log_path(self, base_path):
        nums = [int(name.split("_")[1])
                for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) and name.startswith("YOPO_") and name.split("_")[1].isdigit()]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"YOPO_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard log to ", next_path)
        return next_path
