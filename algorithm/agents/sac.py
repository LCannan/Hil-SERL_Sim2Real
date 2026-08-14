from typing import Dict, Optional, Tuple, FrozenSet, Iterable, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.amp import autocast, GradScaler

from algorithm.networks.actor_critic_nets import (
    GaussianPolicy,
    GraspCritic,
    Critic,
    CriticEnsemble,
)
from algorithm.networks.lagrange import GeqLagrangeMultiplier
from algorithm.networks.mlp import MLP
from algorithm.utils.gripper import (
    GRIPPER_STAY_INDEX,
    NUM_GRIPPER_ACTIONS,
    action_to_index as gripper_action_to_index,
    index_to_action as gripper_index_to_action,
)
from algorithm.vision.timm_encoder import create_encoder
from algorithm.vision.pointnet import PointNetEncoder
from algorithm.common.encoding import EncodingWrapper
from algorithm.utils.torch_utils import dict_apply


class SACAgent:
    """SAC over the continuous action space, optionally with a discrete gripper.

    When ``config["grasp_critic"]`` is set, the last action dimension is split
    off and handled by a separate double-DQN head rather than by the Gaussian
    policy -- the paper's two-MDP formulation (Sec. 3.3, eq. 3), where the two
    share the state space, reward and discount but not the action space.  The
    continuous policy and critic then see only ``action[..., :-1]``, and
    :meth:`sample_actions` reassembles the full-width action so nothing outside
    this class has to know.

    The gripper channel is genuinely categorical -- the envs latch it at +-0.5 --
    so a tanh-squashed Gaussian both wastes its density on a dead zone and, at
    evaluation time, almost never produces a ``mode()`` past the threshold.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        temp: nn.Module,
        encoder: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temp_optimizer: torch.optim.Optimizer,
        encoder_optimizer: torch.optim.Optimizer,
        config: dict,
        grasp_critic: Optional[nn.Module] = None,
        grasp_critic_target: Optional[nn.Module] = None,
        grasp_critic_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.temp = temp
        self.encoder = encoder
        self.grasp_critic = grasp_critic
        self.grasp_critic_target = grasp_critic_target

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.temp_optimizer = temp_optimizer
        self.encoder_optimizer = encoder_optimizer
        self.grasp_critic_optimizer = grasp_critic_optimizer
        self.config = config
        self.device = next(actor.parameters()).device
        self._training = True
        # Running per-class tally for `update_bc`'s weighted cross-entropy; see
        # `_grasp_class_weights` for why it accumulates rather than resetting.
        self._grasp_class_counts = None

        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    @property
    def has_grasp_critic(self) -> bool:
        return self.grasp_critic is not None

    def _continuous_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Drop the gripper column when it is owned by the discrete head."""
        return actions[..., :-1] if self.has_grasp_critic else actions


    def _compute_next_actions(
        self,
        obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_action_distribution = self.actor(obs_enc)
        next_actions, next_actions_log_probs = next_action_distribution.sample_and_log_prob()

        assert next_actions.shape == self._continuous_actions(batch["actions"]).shape
        assert next_actions_log_probs.shape == (batch["actions"].shape[0],)

        return next_actions, next_actions_log_probs


    def _critic_loss_fn(
        self, 
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size = batch["rewards"].shape[0]
        
        with torch.no_grad():
            next_actions, next_actions_log_probs = self._compute_next_actions(next_obs_enc, batch)
            
            target_qs = self.critic_target(next_obs_enc, next_actions)
            
            if self.config["critic_subsample_size"] is not None:
                indices = torch.randperm(self.config["critic_ensemble_size"])
                indices = indices[:self.config["critic_subsample_size"]]
                target_qs = target_qs[indices]
            
            target_q = target_qs.min(dim=0)[0]
            assert target_q.shape == (batch_size,)
            
            # Compute backup
            target = (
                batch["rewards"] + 
                self.config["discount"] * batch["masks"] * target_q
            )
            
            if self.config["backup_entropy"]:
                temperature = self.temp()
                target = target - temperature * next_actions_log_probs

        current_qs = self.critic(obs_enc, self._continuous_actions(batch["actions"]))

        assert current_qs.shape == (self.config["critic_ensemble_size"], batch_size)
        
        critic_loss = F.mse_loss(
            current_qs, 
            target.unsqueeze(0).expand(self.config["critic_ensemble_size"], -1)
        )

        info = {
            "critic_loss": critic_loss.item(),
            "q_values": current_qs.mean().item(),
            "target_q": target.mean().item(),
        }
        
        return critic_loss, info

    def _actor_loss_fn(
        self,
        obs_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        temperature = self.temp().detach()
        dist = self.actor(obs_enc)
        actions, log_probs = dist.sample_and_log_prob()

        q_values = self.critic(obs_enc, actions)
        q_values = q_values.mean(dim=0)

        actor_loss = (temperature * log_probs - q_values).mean()

        info = {
            "actor_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item(),
            "temperature": temperature.item(),
        }

        return actor_loss, info

    def _temperature_loss_fn(
        self, 
        obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        _, next_actions_log_probs = self._compute_next_actions(obs_enc, batch)
        entropy = -next_actions_log_probs.mean()
            
        temperature_loss = self.temp(
            lhs=entropy.detach(),
            rhs=self.config["target_entropy"]
        )
        
        info = {"temperature_loss": temperature_loss.item()}
        return temperature_loss, info

    def _update_critic(self, obs_enc: torch.Tensor, next_obs_enc: torch.Tensor, batch: Dict[str, torch.Tensor]):
        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            critic_loss, critic_info = self._critic_loss_fn(obs_enc, next_obs_enc.detach(), batch)

        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)
        self.scaler.step(self.encoder_optimizer)
        self.scaler.update()
        
        with torch.no_grad():
            tau = self.config["soft_target_update_rate"]
            for target, source in zip(
                self.critic_target.parameters(), 
                self.critic.parameters()
            ):
                target.data.mul_(1 - tau)
                target.data.add_(tau * source.data)
        
        return critic_info

    def _grasp_critic_loss_fn(
        self,
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """Double-DQN loss over the discrete gripper actions -- the paper's eq. (3).

        Actions are chosen with the online network and valued with the target
        one, which is the standard decoupling that stops the max operator from
        compounding its own overestimation.

        The grasp penalty enters *here only*, never in the continuous critic's
        reward.  That follows upstream HIL-SERL: the arm should not be charged
        for the gripper's decisions.  Note the paper's own formulation gives both
        MDPs the same `r` and would imply folding it into both -- it never says
        which, and it never gives a magnitude.
        """
        batch_size = batch["rewards"].shape[0]
        rows = torch.arange(batch_size, device=obs_enc.device)
        grasp_actions = gripper_action_to_index(batch["actions"][..., -1])

        # The online net's forward passes come first, and the target is built
        # from `.detach()` rather than inside a `torch.no_grad()` block.  Under
        # `autocast`, a no_grad forward through a module caches its downcast
        # weights *without* grad, and a later forward through the same module
        # reuses that cache -- so the loss silently comes back with
        # `requires_grad=False` and `backward()` raises "element 0 of tensors
        # does not require grad".  `_critic_loss_fn` above is safe from this only
        # because its target reads `critic_target`, a separate module; double DQN
        # necessarily queries the online net twice.
        predicted_qs = self.grasp_critic(obs_enc)
        next_qs = self.grasp_critic(next_obs_enc)

        with torch.no_grad():
            # Action chosen by the online net, valued by the target net.
            best_next = next_qs.detach().argmax(dim=-1)
            target_next_q = self.grasp_critic_target(next_obs_enc)[rows, best_next]

            grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
            target_q = (
                grasp_rewards
                + self.config["discount"] * batch["masks"] * target_next_q
            )

        predicted_q = predicted_qs[rows, grasp_actions]
        grasp_critic_loss = F.mse_loss(predicted_q, target_q)

        info = {
            "grasp_critic_loss": grasp_critic_loss.item(),
            "grasp_q_values": predicted_q.mean().item(),
            "grasp_target_q": target_q.mean().item(),
            "grasp_penalty": batch["grasp_penalty"].mean().item(),
            # Fraction of the batch the head would actually act on.  If this
            # collapses to 0 the head has degenerated to a constant "stay" and
            # the gripper will never move -- the failure this whole split exists
            # to avoid, and one that no loss value would reveal on its own.
            "grasp_action_nonstay_frac": (
                (predicted_qs.detach().argmax(dim=-1) != GRIPPER_STAY_INDEX)
                .float()
                .mean()
                .item()
            ),
        }
        return grasp_critic_loss, info

    def _update_grasp_critic(
        self,
        obs_enc: torch.Tensor,
        next_obs_enc: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ):
        self.grasp_critic_optimizer.zero_grad()

        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            grasp_loss, grasp_info = self._grasp_critic_loss_fn(
                obs_enc, next_obs_enc, batch
            )

        self.scaler.scale(grasp_loss).backward()
        self.scaler.step(self.grasp_critic_optimizer)
        self.scaler.update()

        with torch.no_grad():
            tau = self.config["soft_target_update_rate"]
            for target, source in zip(
                self.grasp_critic_target.parameters(),
                self.grasp_critic.parameters(),
            ):
                target.data.mul_(1 - tau)
                target.data.add_(tau * source.data)

        return grasp_info

    def _update_actor(self, obs_enc: torch.Tensor):
        self.actor_optimizer.zero_grad()

        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            actor_loss, actor_info = self._actor_loss_fn(obs_enc.detach())

        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        return actor_info

    def _update_temperature(self, next_obs_enc: torch.Tensor, batch: Dict[str, torch.Tensor]):
        self.temp_optimizer.zero_grad()
        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            temp_loss, temp_info = self._temperature_loss_fn(next_obs_enc.detach(), batch)
        self.scaler.scale(temp_loss).backward()
        self.scaler.step(self.temp_optimizer)
        self.scaler.update()

        return temp_info

    def update_bc(self, batch: Dict[str, torch.Tensor], std_target: float = 0.1) -> Dict:
        """Supervised update: regress the policy's mode onto demonstrated actions.

        This is the only way to make an untrained policy stop flailing.  Shrinking
        the exploration noise alone does not do it: measured on this repo's
        networks, driving ``std_max`` from 5.0 down to 0.1 takes ``P(|a| > 0.9)``
        from 26% to 1.6% but leaves ``mean|mode|`` at 0.413 throughout -- the
        deterministic output is whatever the randomly initialised mean layer
        happens to emit, and only a gradient on that layer moves it.

        Both halves of the distribution have to be trained, and this was
        measured rather than assumed.  Fitting the mean alone drops
        ``mean|mode|`` to 0.03 while leaving ``scale`` at its initialised 0.2-2.3
        -- so ``sample()`` still saturates tanh and the arm still thrashes, which
        looks from the outside as though the pretraining did nothing.  The std
        term below is what makes the *sampled* action follow the mean.

        MSE on ``dist.mode()`` rather than a log-likelihood, because
        ``TanhNormal.log_prob`` runs ``atanh(clamp(a, +-(1 - 1e-7)))`` and a
        demonstrated action sitting on +-1 is exactly that singularity.  With a
        grasp critic configured the offending channel is no longer part of this
        distribution at all -- the Cartesian channels peak around 0.25 -- but the
        mode regression is kept because it is what the std term above is paired
        with, and it costs nothing.

        With a grasp critic, the gripper is warm-started here too, by a weighted
        cross-entropy on its logits.  Without that term the pretrain would fit
        only the continuous dims and leave the discrete head at its random
        initialisation, which under pure-argmax action selection means an
        arbitrary but *constant* gripper command for the whole first rollout.
        Fitting a Q-network as a classifier is not the Bellman objective it will
        be trained on online; it only has to make the head non-degenerate for the
        first few minutes of teleoperation.  See ``_grasp_class_weights`` for the
        class-imbalance handling.

        Only the actor and the grasp critic are trained.  The encoder keeps its
        ImageNet weights: with no critic signal yet there is nothing to tell it
        which features matter, and letting BC reshape it tends to destroy the
        pretrained representation that the online phase then has to rebuild.
        """
        batch = dict_apply(batch, lambda x: x.to(self.device))
        augmentation_function = self.config.get("augmentation_function")
        if augmentation_function is not None:
            aug_seed = torch.randint(0, 2**31, (1,)).item()
            batch = augmentation_function(batch, aug_seed, self.device)

        self.actor_optimizer.zero_grad()
        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            with torch.no_grad():
                obs_enc = self.encoder(batch["observations"])
            dist = self.actor(obs_enc)
            mode_loss = F.mse_loss(
                dist.mode(), self._continuous_actions(batch["actions"])
            )
            # Regress log-std onto a small constant rather than maximising a
            # likelihood: it leaves the exploration scale somewhere the online
            # phase can still widen, without the atanh singularity above.
            std_loss = F.mse_loss(
                dist.scale.log(),
                torch.full_like(dist.scale, math.log(std_target)),
            )
            bc_loss = mode_loss + std_loss

        self.scaler.scale(bc_loss).backward()
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        info = {
            "bc_loss": bc_loss.item(),
            "bc_mode_loss": mode_loss.item(),
            "bc_std_loss": std_loss.item(),
        }

        if self.has_grasp_critic:
            self.grasp_critic_optimizer.zero_grad()
            with autocast(self.device.type, enabled=self.device.type == "cuda"):
                grasp_actions = gripper_action_to_index(batch["actions"][..., -1])
                grasp_logits = self.grasp_critic(obs_enc)
                grasp_bc_loss = F.cross_entropy(
                    grasp_logits,
                    grasp_actions,
                    weight=self._grasp_class_weights(grasp_actions),
                )

            self.scaler.scale(grasp_bc_loss).backward()
            self.scaler.step(self.grasp_critic_optimizer)
            self.scaler.update()

            with torch.no_grad():
                predicted = grasp_logits.detach().argmax(dim=-1)
                info.update(
                    {
                        "bc_grasp_loss": grasp_bc_loss.item(),
                        "bc_grasp_accuracy": (
                            (predicted == grasp_actions).float().mean().item()
                        ),
                        # Batch accuracy is close to meaningless here -- at a
                        # 1.4% non-stay rate most batches contain none at all, so
                        # predicting "stay" everywhere scores ~1.0 honestly.  This
                        # is the number to watch instead: 0 means the head has
                        # collapsed to a constant and the gripper will never move.
                        "bc_grasp_nonstay_frac": (
                            (predicted != GRIPPER_STAY_INDEX).float().mean().item()
                        ),
                    }
                )
            # Keep the target in step, or the first online DQN updates bootstrap
            # off a random network and undo the warm start immediately.
            self.grasp_critic_target.load_state_dict(self.grasp_critic.state_dict())

        return info

    def _grasp_class_weights(self, grasp_actions: torch.Tensor) -> torch.Tensor:
        """Inverse-frequency class weights from a *running* count, not this batch.

        The gripper classes are extremely unbalanced -- measured on the
        ``pick_place_milk`` demos, 98.56% of transitions are "stay" (8469 of
        8593, against 55 close and 69 open) -- and the failure mode is the head
        collapsing to a constant "stay", which under argmax action selection
        means a gripper that never moves.

        Weighting buys a faster and more reliable escape from that basin rather
        than being strictly necessary: measured on a synthetic task at this class
        ratio, at a 100-step budget unweighted collapsed in 1 of 3 seeds while
        weighted converged in 3 of 3, and by 200 steps both converged.  Cheap
        insurance for a short ``--pretrain_steps``.

        The count accumulates rather than being taken per batch, and that part
        does matter.  At batch size 64 roughly 63% of batches contain no "open"
        sample at all, so per-batch weights alternate between ignoring a class
        and hitting it with a ~60x gradient spike relative to "stay".  A running
        count approaches the dataset prior and holds steady, so a rare sample
        carries the same corrective weight whichever batch it lands in.
        """
        counts = torch.bincount(grasp_actions, minlength=NUM_GRIPPER_ACTIONS).float()
        if self._grasp_class_counts is None:
            self._grasp_class_counts = counts
        else:
            self._grasp_class_counts = self._grasp_class_counts + counts
        seen = self._grasp_class_counts
        # Laplace-smoothed, so a class not yet seen gets a large but finite
        # weight rather than dividing by zero.
        return seen.sum() / (NUM_GRIPPER_ACTIONS * (seen + 1.0))

    def state_dict(self) -> dict:
        serializable_config = {k: v for k, v in self.config.items()
                          if not callable(v)}

        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "temp": self.temp.state_dict(),
            "encoder": self.encoder.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temp_optimizer": self.temp_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "config": serializable_config,
        }
        # Must be here for the actor to ever receive a trained gripper policy:
        # `state_dict_to_numpy` walks whatever this returns, and the broadcast
        # carries only what it finds.
        if self.has_grasp_critic:
            state["grasp_critic"] = self.grasp_critic.state_dict()
            state["grasp_critic_target"] = self.grasp_critic_target.state_dict()
            state["grasp_critic_optimizer"] = self.grasp_critic_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.actor.load_state_dict(state_dict["actor"], strict=strict)
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        self.critic_target.load_state_dict(state_dict["critic_target"], strict=strict)
        self.temp.load_state_dict(state_dict["temp"], strict=strict)
        self.encoder.load_state_dict(state_dict["encoder"], strict=strict)

        # Guarded, unlike the five above: the actor's broadcast callback and the
        # checkpoint resume both pass strict=False and expect a partial dict, and
        # `state_dict_to_numpy` strips the optimizer entries before broadcast.
        # A missing key here is normal, not an error.
        if self.has_grasp_critic:
            if "grasp_critic" in state_dict:
                self.grasp_critic.load_state_dict(
                    state_dict["grasp_critic"], strict=strict
                )
            if "grasp_critic_target" in state_dict:
                self.grasp_critic_target.load_state_dict(
                    state_dict["grasp_critic_target"], strict=strict
                )
            if "grasp_critic_optimizer" in state_dict:
                self.grasp_critic_optimizer.load_state_dict(
                    state_dict["grasp_critic_optimizer"]
                )

        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        if "critic_optimizer" in state_dict:
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if "temp_optimizer" in state_dict:
            self.temp_optimizer.load_state_dict(state_dict["temp_optimizer"])
        if "encoder_optimizer" in state_dict:
            self.encoder_optimizer.load_state_dict(state_dict["encoder_optimizer"])
        if "config" in state_dict:
            self.config.update(state_dict["config"])

    def to(self, device: torch.device) -> "SACAgent":
        device = torch.device(device) if isinstance(device, str) else device
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.critic_target = self.critic_target.to(device)
        self.temp = self.temp.to(device)
        self.encoder = self.encoder.to(device)
        if self.has_grasp_critic:
            self.grasp_critic = self.grasp_critic.to(device)
            self.grasp_critic_target = self.grasp_critic_target.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True) -> "SACAgent":
        self._training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(False)
        self.temp.train(mode)
        self.encoder.train(mode)
        if self.has_grasp_critic:
            self.grasp_critic.train(mode)
            self.grasp_critic_target.train(False)
        return self

    def eval(self) -> "SACAgent":
        return self.train(False)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"})
    ) -> Dict:
        batch = dict_apply(batch, lambda x: x.to(self.device))
        # Apply data augmentation
        if self.config.get("augmentation_function") is not None:
            aug_seed = torch.randint(0, 2**31, (1,)).item()
            batch = self.config["augmentation_function"](batch, aug_seed, self.device)

        reward_bias = self.config.get("reward_bias", 0.0)
        if reward_bias != 0.0:
            batch = {**batch, "rewards": batch["rewards"] + reward_bias}
        
        info = {}

        obs_enc = self.encoder(batch["observations"])
        next_obs_enc = self.encoder(batch["next_observations"])

        # Update critic
        if "critic" in networks_to_update:
            critic_info = self._update_critic(obs_enc, next_obs_enc, batch)
            info.update(critic_info)

        # Update grasp critic.  Both encodings are detached, unlike the
        # continuous critic's: `_update_critic` above lets its gradient reach the
        # encoder and steps `encoder_optimizer`, and adding a second contribution
        # here would step that optimizer twice per iteration on a stale graph.
        # A deliberate divergence from upstream, whose single optimizer tree
        # makes the question moot.
        if "grasp_critic" in networks_to_update and self.has_grasp_critic:
            grasp_info = self._update_grasp_critic(
                obs_enc.detach(), next_obs_enc.detach(), batch
            )
            info.update(grasp_info)

        # Update actor
        if "actor" in networks_to_update:
            actor_info = self._update_actor(obs_enc.detach())
            info.update(actor_info)

        # Update temperature
        if "temperature" in networks_to_update:
            temp_info = self._update_temperature(next_obs_enc.detach(), batch)
            info.update(temp_info)

        return info

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Dict[str, torch.Tensor],
        argmax: bool = False
    ) -> torch.Tensor:
        observations = dict_apply(observations, lambda x: x.to(self.device))
        obs_enc = self.encoder(observations)
        dist = self.actor(obs_enc)
        actions = dist.mode() if argmax else dist.sample()

        if not self.has_grasp_critic:
            return actions

        # Greedy in both training and evaluation -- the paper takes the argmax
        # "at training or inference time" and never mentions epsilon-greedy.
        # Exploration for this head comes from the human's interventions and the
        # demonstrations, not from action noise.
        grasp_index = self.grasp_critic(obs_enc).argmax(dim=-1)
        grasp_action = gripper_index_to_action(grasp_index).to(actions.dtype)
        return torch.cat([actions, grasp_action.unsqueeze(-1)], dim=-1)

    @classmethod
    def create_pixels(
        cls,
        sample_obs: Dict[str, torch.Tensor],
        sample_action: torch.Tensor,
        encoder_type: str = "resnet18-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = None,
        policy_network_kwargs: dict = None,
        policy_kwargs: dict = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1e-2,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[Callable] = None,
        reward_bias: float = 0.0,
        image_size: Tuple[int, int] = (128, 128),
        grasp_critic: bool = False,
        grasp_critic_network_kwargs: dict = None,
        **kwargs,
    ) -> "SACAgent":

        image_keys = tuple(image_keys)
        
        # Default kwargs
        if critic_network_kwargs is None:
            critic_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_network_kwargs is None:
            policy_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_kwargs is None:
            policy_kwargs = {
                "tanh_squash_distribution": True,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
            }
        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}
        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}
        if grasp_critic_network_kwargs is None:
            # The paper's tables give 256x256 for the grasp critic in every
            # gripper task, the same width as the motion policy.
            grasp_critic_network_kwargs = {"hidden_dims": [256, 256]}
        grasp_critic_network_kwargs = {
            **grasp_critic_network_kwargs, "activate_final": True
        }

        action_dim = sample_action.shape[-1]
        # With a discrete gripper head the last dimension leaves the continuous
        # policy and critic entirely -- they are the paper's M_1, over A_1 only.
        continuous_action_dim = action_dim - 1 if grasp_critic else action_dim
        if grasp_critic and continuous_action_dim < 1:
            raise ValueError(
                "grasp_critic needs an action space wider than the gripper "
                f"channel alone, got action_dim={action_dim}"
            )

        encoders = create_encoder(
            encoder_type=encoder_type,
            image_keys=image_keys,
            image_size=image_size,
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=8,
            bottleneck_dim=256,
        )
        encoder_def = EncodingWrapper(
            encoder=encoders,
            state_dim=sample_obs["state"].shape[-1],
            use_proprio=use_proprio,
            proprio_latent_dim=64,
            enable_stacking=True,
            image_keys=image_keys,
        )

        # Create policy network
        encoder_output_dim = encoder_def.output_dim
        policy_hidden_dims = [encoder_output_dim] + policy_network_kwargs.get("hidden_dims", [256, 256])
        policy_network = MLP(
            hidden_dims=policy_hidden_dims,
            activate_final=True,
            use_layer_norm=policy_network_kwargs.get("use_layer_norm", False),
            activations=policy_network_kwargs.get("activation", nn.Tanh()),
        )
        actor = GaussianPolicy(
            network=policy_network,
            action_dim=continuous_action_dim,
            **policy_kwargs,
        )

        # Create critics
        critic_hidden_dims = [encoder_output_dim + continuous_action_dim] + critic_network_kwargs.get("hidden_dims", [256, 256])
        critics = []
        for _ in range(critic_ensemble_size):
            critic_network = MLP(
                hidden_dims=critic_hidden_dims,
                activate_final=True,
                use_layer_norm=critic_network_kwargs.get("use_layer_norm", False),
                activations=critic_network_kwargs.get("activation", nn.Tanh()),
            )

            critics.append(Critic(network=critic_network))

        critic = CriticEnsemble(critics)
        critic_target = deepcopy(critic)

        # The discrete gripper head, Q(s, .) over {close, stay, open}.  Takes no
        # action input, so its trunk starts from the encoding alone.
        grasp_critic_net = None
        grasp_critic_target = None
        grasp_critic_optimizer = None
        if grasp_critic:
            grasp_hidden_dims = [encoder_output_dim] + grasp_critic_network_kwargs.get(
                "hidden_dims", [256, 256]
            )
            grasp_critic_net = GraspCritic(
                network=MLP(
                    hidden_dims=grasp_hidden_dims,
                    activate_final=True,
                    use_layer_norm=grasp_critic_network_kwargs.get(
                        "use_layer_norm", False
                    ),
                    activations=grasp_critic_network_kwargs.get(
                        "activation", nn.Tanh()
                    ),
                ),
                output_dim=NUM_GRIPPER_ACTIONS,
            )
            grasp_critic_target = deepcopy(grasp_critic_net)

        # Create temperature (Lagrange multiplier)
        temp = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )
        # Set target entropy.  Sized to the continuous action space, so splitting
        # the gripper out moves it from -3.5 to -3.0 for a 7-dim task.
        target_entropy = kwargs.get("target_entropy")
        if target_entropy is None:
            target_entropy = -continuous_action_dim / 2

        # Build config with pixel-specific fields
        config_kwargs = {
            "discount": kwargs.get("discount", 0.97),
            "soft_target_update_rate": kwargs.get("soft_target_update_rate", 0.005),
            "target_entropy": target_entropy,
            "backup_entropy": kwargs.get("backup_entropy", False),
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            "image_keys": image_keys,
            "augmentation_function": augmentation_function,
            "reward_bias": reward_bias,
            "grasp_critic": grasp_critic,
        }

        # Create optimizers
        temp_optimizer = torch.optim.Adam(temp.parameters(), lr=3e-4)
        encoder_optimizer = torch.optim.Adam(encoder_def.parameters(), lr=3e-4)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
        if grasp_critic:
            grasp_critic_optimizer = torch.optim.Adam(
                grasp_critic_net.parameters(), lr=3e-4
            )

        agent = cls(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            temp=temp,
            encoder=encoder_def,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temp_optimizer=temp_optimizer,
            encoder_optimizer=encoder_optimizer,
            config=config_kwargs,
            grasp_critic=grasp_critic_net,
            grasp_critic_target=grasp_critic_target,
            grasp_critic_optimizer=grasp_critic_optimizer,
        )

        return agent

    @classmethod
    def create_pointcloud(
        cls,
        sample_obs: Dict[str, torch.Tensor],
        sample_action: torch.Tensor,
        encoder_type: str = "pointnet",
        use_proprio: bool = False,
        critic_network_kwargs: dict = None,
        policy_network_kwargs: dict = None,
        policy_kwargs: dict = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1e-2,
        image_keys: Iterable[str] = ("point_cloud",),
        augmentation_function: Optional[Callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ) -> "SACAgent":

        image_keys = tuple(image_keys)
        
        # Default kwargs
        if critic_network_kwargs is None:
            critic_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_network_kwargs is None:
            policy_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_kwargs is None:
            policy_kwargs = {
                "tanh_squash_distribution": True,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
            }
        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}
        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}
        
        action_dim = sample_action.shape[-1]
        
        encoders = {}
        for image_key in image_keys:
            point_channels = sample_obs[image_key].shape[-1]
            if encoder_type == "pointnet":
                encoders[image_key] = PointNetEncoder(point_channels, 64)

        encoder_def = EncodingWrapper(
            encoder=encoders,
            state_dim=sample_obs["state"].shape[-1],
            use_proprio=use_proprio,
            proprio_latent_dim=64,
            enable_stacking=True,
            image_keys=image_keys,
        )

        # Create policy network
        encoder_output_dim = encoder_def.output_dim
        policy_hidden_dims = [encoder_output_dim] + policy_network_kwargs.get("hidden_dims", [256, 256])
        policy_network = MLP(
            hidden_dims=policy_hidden_dims,
            activate_final=True,
            use_layer_norm=policy_network_kwargs.get("use_layer_norm", False),
            activations=policy_network_kwargs.get("activation", nn.Tanh()),
        )
        actor = GaussianPolicy(
            network=policy_network,
            action_dim=action_dim,
            **policy_kwargs,
        )
        
        # Create critics
        critic_hidden_dims = [encoder_output_dim + action_dim] + critic_network_kwargs.get("hidden_dims", [256, 256])
        critics = []
        for _ in range(critic_ensemble_size):
            critic_network = MLP(
                hidden_dims=critic_hidden_dims,
                activate_final=True,
                use_layer_norm=critic_network_kwargs.get("use_layer_norm", False),
                activations=critic_network_kwargs.get("activation", nn.Tanh()),
            )
            
            critics.append(Critic(network=critic_network))
        
        critic = CriticEnsemble(critics)
        critic_target = deepcopy(critic)
        
        # Create temperature (Lagrange multiplier)
        temp = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )
        # Set target entropy
        target_entropy = kwargs.get("target_entropy")
        if target_entropy is None:
            target_entropy = -action_dim / 2
        
        # Build config with pixel-specific fields
        config_kwargs = {
            "discount": kwargs.get("discount", 0.97),
            "soft_target_update_rate": kwargs.get("soft_target_update_rate", 0.005),
            "target_entropy": target_entropy,
            "backup_entropy": kwargs.get("backup_entropy", False),
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            "image_keys": image_keys,
            "augmentation_function": augmentation_function,
            "reward_bias": reward_bias,
        }
    
        # Create optimizers
        temp_optimizer = torch.optim.Adam(temp.parameters(), lr=3e-4)
        encoder_optimizer = torch.optim.Adam(encoder_def.parameters(), lr=3e-4)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
        
        agent = cls(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            temp=temp,
            encoder=encoder_def,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temp_optimizer=temp_optimizer,
            encoder_optimizer=encoder_optimizer,
            config=config_kwargs,
        )
        
        return agent
