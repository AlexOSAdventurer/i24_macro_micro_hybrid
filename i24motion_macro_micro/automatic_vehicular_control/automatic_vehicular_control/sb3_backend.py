"""
Stable-Baselines3 backend for AVC's single-agent scenarios (so far the ring), selected by backend='sb3' (the default).

Main.run() hands training here instead of to the Ray rollout workers and the hand-written optimizers in ut.py, and
Main.eval() loads the resulting checkpoints through load_policy(). The rest of AVC is unchanged: the environments,
their per-worker reward normalization (NormEnv), worker_kwargs, the rollout layout (each worker starts one episode of
horizon + skip_stat_steps controlled steps per iteration), the network shape and initialization, the hyperparameter
defaults, and the run directory (config.yaml, train_results.csv, models/).

AVC's TRPO settings map onto sb3-contrib's TRPO flags:
    max_kl (start_max_kl == end_max_kl)  -> target_kl
    steps_cg                             -> cg_max_steps
    damping                              -> cg_damping
    steps_backtrack                      -> line_search_max_iter, with line_search_shrinking_factor=0.5 (AVC halves)
    adv_norm                             -> normalize_advantage
    gamma, lam                           -> gamma, gae_lambda
    n_gds, n_minibatches                 -> n_critic_updates, batch_size
    lr, lr_schedule                      -> learning_rate (it only drives the critic, as in AVC)
    opt, betas, l2                       -> the policy's optimizer_class / optimizer_kwargs
    n_rollouts_per_step                  -> number of environments
    horizon + skip_stat_steps            -> the episode length (and n_steps when episodes_per_update is 1)
    episodes_per_update                  -> n_steps = episodes_per_update * (horizon + skip_stat_steps)
    n_steps (iterations)                 -> total_timesteps = n_steps * n_envs * (horizon + skip_stat_steps)
    layers                               -> net_arch=dict(pi=..., vf=...), activation_fn
    use_critic=False                     -> value net zeroed and never trained (n_critic_updates=0), so V = 0 and
                                            GAE(lam=1) is AVC's plain discounted return

Differences from AVC's own TRPO, accepted because the controllers are retrained rather than reproduced bit for bit:
    - SB3's Gaussian has a state-independent log_std parameter (starting at AVC's initial std of 1) instead of a
      log_std output of the policy head.
    - SB3 stores the unclipped sampled action (the environment gets the clipped one); AVC stored the clipped action.
    - sb3-contrib's line search accepts any surrogate improvement with KL(new || old) < target_kl; AVC also required
      accept_ratio of the linearly predicted improvement and bounded KL(old || new). sb3-contrib's conjugate
      gradient starts from 1e-4 * randn and stops at a 1e-10 residual; AVC's started at 0.
    - A worker whose episode ends early (collision) is reset at the start of the next iteration, which truncates its
      last episode at the buffer end; AVC kept that worker collecting until it had a full-length episode.
    - n_workers is not used: each of the n_rollouts_per_step environments runs in its own process.

Why episodes_per_update exists
------------------------------
AVC ran one 10,000-step episode per worker per update, so the rollout boundary was
the episode boundary.  Shortening the ring's episodes to match the I-24 corridor's
bubble traversal (~380 steps at 0.1 s) would otherwise shrink the batch by the same
factor and turn 50 trust-region steps into thousands, which is a different
optimiser rather than a different road.  Setting episodes_per_update keeps the
paper's batch with short episodes: n_steps stays an exact multiple of the episode
length, so no episode is ever cut at a buffer boundary -- which matters without a
critic, where a cut episode's return gets no bootstrap -- and every worker resets
on the same step, so a reset stalls the synchronous fleet once rather than once per
worker.
"""
import functools
import os
import re
from time import time

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from automatic_vehicular_control.u import Namespace
from automatic_vehicular_control.ut import FFN, TRPO as LegacyTRPO
from automatic_vehicular_control.env import GymnasiumEnv

ACTIVATIONS = dict(tanh=nn.Tanh, relu=nn.ReLU, sigmoid=nn.Sigmoid)
NETWORK_DEFAULTS = dict(layers=[64, 'tanh', 64, 'tanh'], weight_scale='default', weight_init='orthogonal') # ut.FFN's

def apply_defaults(c):
    """Fill c with the defaults the legacy code path uses, and refuse settings sb3-contrib's TRPO can't express"""
    alg = c.alg if isinstance(c.alg, str) else c.alg.__name__
    if alg != 'TRPO':
        raise NotImplementedError(f"backend='sb3' implements alg='TRPO' only, got {alg!r}; backend='legacy' runs the hand-written one")
    LegacyTRPO(c) # only sets AVC's TRPO and Algorithm defaults on c: use_critic, max_kl, steps_cg, damping, n_gds, lam, ...
    # episodes_per_update 1 is AVC's own layout: one episode per worker per update
    c.setdefaults(**NETWORK_DEFAULTS, lr=1e-4, adv_norm=False, episodes_per_update=1) # lr is Main._lr's default
    if c.start_max_kl != c.end_max_kl:
        raise NotImplementedError("sb3-contrib's target_kl is constant; set max_kl rather than start_max_kl / end_max_kl")
    if c.normclip:
        raise NotImplementedError("sb3-contrib's TRPO critic update does not clip gradients (normclip)")
    if c.get('last_unbiased'):
        raise NotImplementedError('last_unbiased edits the legacy FFN policy head')
    return c

class Schedule:
    """Main.schedule() as a function of SB3's progress_remaining, which runs 1 -> 0 like AVC's frac_left"""
    def __init__(self, coef, schedule):
        self.coef, self.schedule = coef, schedule

    def __call__(self, frac_left):
        if self.schedule == 'linear':
            return self.coef * frac_left
        if self.schedule == 'cosine':
            return self.coef * (np.cos((1 - frac_left) * np.pi) + 1) / 2
        raise ValueError(f'unknown lr_schedule {self.schedule!r}')

def policy_kwargs(c):
    """SB3 ActorCriticPolicy arguments for AVC's network: c.layers for both heads, no shared layers, AVC's optimizer"""
    layers = c.layers
    if not isinstance(layers, (list, tuple)):
        raise NotImplementedError(f'only list layer specs (used by both the policy and value heads) are supported, got {layers!r}')
    sizes, activations = list(layers[0::2]), list(layers[1::2])
    if not (all(isinstance(s, (int, np.integer)) for s in sizes) and len(activations) == len(sizes) and len(set(activations)) <= 1):
        raise NotImplementedError(f'SB3 applies one activation after every hidden layer, which cannot express {layers!r}')
    opt = c.get('opt', 'Adam')
    if opt == 'Adam': # eps is explicit because SB3 substitutes eps=1e-5 when optimizer_kwargs is None; AVC used torch's 1e-8
        optimizer_class, optimizer_kwargs = th.optim.Adam, dict(betas=tuple(c.get('betas', (0.9, 0.999))), eps=1e-8, weight_decay=c.get('l2', 0))
    elif opt == 'RMSprop':
        optimizer_class, optimizer_kwargs = th.optim.RMSprop, dict(weight_decay=c.get('l2', 0))
    else:
        raise ValueError(f'unknown opt {opt!r}')
    sizes = [int(s) for s in sizes]
    return dict(
        net_arch=dict(pi=sizes, vf=sizes),
        activation_fn=ACTIVATIONS[activations[0]] if activations else nn.Tanh,
        ortho_init=False, # avc_init replaces SB3's initialization
        log_std_init=0., # AVC's policy head starts with log_std ~ 0 (its output layer has gain 0.01)
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
    )

def avc_init(policy, c):
    """
    Initialize SB3's networks the way ut.FFN.sequential_init initializes AVC's: c.weight_init (orthogonal) with gain 1
    on hidden layers and 0.01 on each head's output layer (c.weight_scale='default'), zero biases. SB3's ortho_init
    uses sqrt(2) / 0.01 / 1 instead. Without a critic the value net is zeroed, and since it is never trained V = 0.
    """
    holder = Namespace(c=c)
    FFN.sequential_init(holder, nn.Sequential(*policy.mlp_extractor.policy_net, policy.action_net), 'policy')
    FFN.sequential_init(holder, nn.Sequential(*policy.mlp_extractor.value_net, policy.value_net), 'value')
    if not c.use_critic:
        nn.init.zeros_(policy.value_net.weight)
        nn.init.zeros_(policy.value_net.bias)

def make_env(main_cls, kwargs, i_worker):
    """One rollout worker's environment, configured the way Main.run configured each Ray worker"""
    c = main_cls(**kwargs, i_worker=i_worker)
    return GymnasiumEnv(c, c.create_env())

def build_vec_env(c):
    n_envs = c.n_rollouts_per_step
    worker_kwargs = c.get('worker_kwargs') or [{}]
    assert n_envs % len(worker_kwargs) == 0, 'n_rollouts_per_step must be a multiple of len(worker_kwargs)'
    n_repeats = n_envs // len(worker_kwargs)
    # As in Main.run's Ray setup, each worker gets the whole config overridden by its worker_kwargs entry. Private
    # entries holding objects (environment, model, ...) are left out; plain ones such as _n_obs are needed
    plain = (bool, int, float, str, list, tuple, dict, type(None))
    base = {k: v for k, v in c.items() if not k.startswith('_') or isinstance(v, plain)}
    env_fns = [functools.partial(make_env, type(c), {**base, 'main': False, 'device': 'cpu', **worker_kwargs[i // n_repeats]}, i)
               for i in range(n_envs)]
    if n_envs > 1 and c.get('subproc', True):
        return SubprocVecEnv(env_fns, start_method='spawn') # spawn: workers inherit none of the parent's torch or socket state
    return DummyVecEnv(env_fns)

def build_model(c, env):
    episode_steps = c.horizon + c.skip_stat_steps
    n_steps = episode_steps * int(c.episodes_per_update)
    model = TRPO(
        ActorCriticPolicy, env,
        learning_rate=Schedule(c.lr, c.lr_schedule) if c.get('lr_schedule') else c.lr,
        n_steps=n_steps,
        batch_size=env.num_envs * n_steps // (c.get('n_minibatches') or 1),
        gamma=c.gamma,
        cg_max_steps=c.steps_cg,
        cg_damping=c.damping,
        line_search_shrinking_factor=0.5,
        line_search_max_iter=c.steps_backtrack,
        n_critic_updates=c.n_gds if c.use_critic else 0,
        gae_lambda=c.lam,
        normalize_advantage=bool(c.adv_norm),
        target_kl=c.start_max_kl,
        policy_kwargs=policy_kwargs(c),
        seed=c.get('seed'),
        device=c.device,
    )
    avc_init(model.policy, c)
    return model

def checkpoints(c):
    """{number of updates: path} of the SB3 checkpoints in c.res/models"""
    found = {}
    for path in c.models.glob('model-*.zip'):
        match = re.search(r'model-(\d+)\.zip$', path)
        if match:
            found[int(match.group(1))] = path
    return found

class AVCCallback(BaseCallback):
    """
    Keeps AVC's per-iteration layout and bookkeeping on top of SB3's loop (an iteration is one rollout and one update):
    - every iteration starts each worker on a fresh episode, as AVC's rollout workers did; SB3 would otherwise carry an
      episode that a collision cut short across the update,
    - AVC's episode stats (speed, speed_rl, collisions, ...) are averaged over each iteration's episodes, logged under
      avc/, and written with the update's statistics as one row of train_results.csv,
    - the model is saved to models/model-{i}.zip after every c.step_save updates, as models/model-{i}.pth was.
    """
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.pending = None

    @property
    def iterations(self):
        """Number of updates done (each rollout is followed by exactly one update)"""
        return self.model.num_timesteps // (self.model.n_envs * self.model.n_steps)

    def save(self):
        path = self.c.models / f'model-{self.iterations}.zip'
        self.model.save(path)
        self.c.log(f'Saved model {path} at step {self.iterations}')

    def _on_training_start(self):
        self.episode_steps = np.zeros(self.training_env.num_envs, dtype=int)
        if self.iterations == 0:
            self.save()

    def _on_rollout_start(self):
        self.flush()
        for i in np.flatnonzero(self.episode_steps):
            obs, _ = self.training_env.env_method('reset', indices=[int(i)])[0]
            self.model._last_obs[i] = obs
            self.model._last_episode_starts[i] = True
        self.episode_steps[:] = 0
        self.rollout_start = time()
        self.episode_stats, self.reward_sum, self.raw_reward_sum, self.n_rewards = [], 0., 0., 0

    def _on_step(self):
        dones = self.locals['dones']
        self.episode_steps += 1
        self.episode_steps[dones] = 0
        self.reward_sum += float(np.sum(self.locals['rewards']))
        self.n_rewards += len(dones)
        for info in self.locals['infos']:
            self.raw_reward_sum += info['raw_reward']
            if 'avc_stats' in info:
                self.episode_stats.append(info['avc_stats'])
        return True

    def _on_rollout_end(self):
        stats = pd.DataFrame(self.episode_stats).mean(numeric_only=True).to_dict() if self.episode_stats else {}
        self.pending = dict(
            rollout_time=time() - self.rollout_start,
            n_episodes=len(self.episode_stats),
            reward_mean=self.reward_sum / self.n_rewards, # the normalized reward the update sees, as AVC logged it
            raw_reward_mean=self.raw_reward_sum / self.n_rewards,
            **stats,
        )
        for k, v in self.pending.items():
            self.logger.record('avc/' + k, v)

    def _on_training_end(self):
        self.flush()

    def flush(self):
        """After an update: write its row of train_results.csv and save the checkpoint if due"""
        if self.pending is None:
            return
        c = self.c
        i = self.iterations
        train_stats = {k.split('/', 1)[1]: v for k, v in self.logger.name_to_value.items() if k.startswith('train/')}
        row = dict(self.pending, **train_stats, total_time=time() - c._run_start_time)
        for k, v in row.items():
            if k not in c._results:
                c._results[k] = np.nan
            c._results.loc[i - 1, k] = v
        c.save_train_results(c._results)
        self.pending = None
        if (c.step_save and i % c.step_save == 0) or i >= c.n_steps: # ring.py sets step_save = 0 when n_steps < 10
            self.save()

def train(c):
    """Main.train for backend='sb3'. Resumes from the latest models/model-*.zip if there is one, as AVC did"""
    apply_defaults(c)
    env = build_vec_env(c)
    saved = checkpoints(c)
    start = max(saved) if saved else 0
    try:
        if saved:
            model = TRPO.load(saved[start], env=env, device=c.device)
            c.log(f'Resuming from {saved[start]}')
            # TRPO.load does not reseed, so a run seeded from another run's checkpoint (v5 from v4) would ignore
            # seed=N. set_random_seed also queues env seeds N+i for the reset that learn() performs. No-op if unset.
            if c.get('seed') is not None:
                model.set_random_seed(c.seed)
        else:
            model = build_model(c, env)
        results = c.load_train_results() if start else None
        c._results = results.loc[:start - 1] if results is not None else pd.DataFrame(index=pd.Series(name='step'))
        c._run_start_time = time() - (c._results['total_time'].iloc[-1] if len(c._results) and 'total_time' in c._results else 0)
        c.try_save_commit(type(c).flow_base)
        model.set_logger(configure(c.res, ['stdout', 'csv'] + (['tensorboard'] if c.tb else [])))
        if start < c.n_steps:
            model.learn(
                total_timesteps=(c.n_steps - start) * env.num_envs * model.n_steps,
                callback=AVCCallback(c), log_interval=1, reset_num_timesteps=not saved,
            )
    finally:
        env.close()
    return model

class PolicyAdapter:
    """An SB3 policy behind ut.FFN's call signature, so that Main.rollout and Main.on_rollout_end run unchanged"""
    def __init__(self, policy):
        self.policy = policy
        policy.set_training_mode(False)

    def __call__(self, obs, value=False, policy=False, argmax=None):
        obs = th.as_tensor(obs, dtype=th.float32, device=self.policy.device)
        single = obs.dim() == len(self.policy.observation_space.shape)
        if single:
            obs = obs[None]
        pred = Namespace()
        with th.no_grad():
            if value:
                pred.value = self.policy.predict_values(obs).view(-1)
            if argmax is not None: # Main.rollout passes argmax=False: AVC's evaluations sample from the policy
                action = self.policy.get_distribution(obs).get_actions(deterministic=argmax)
                pred.action = action[0] if single else action
        return pred

def load_policy(c):
    """Main.eval's model for backend='sb3': models/model-{c.e}.zip when c.e is an update count, else the .zip at path c.e"""
    c.setdefaults(**NETWORK_DEFAULTS)
    path = checkpoints(c).get(c.e) if isinstance(c.e, int) else c.e
    if path and os.path.exists(path):
        policy = TRPO.load(path, device=c.device).policy
        step = c.e
    else: # like the legacy set_state, which left the model untrained when no checkpoint existed (e.g. the av=0 baselines)
        c.log(f'No checkpoint for e={c.e}, evaluating an untrained policy')
        policy = ActorCriticPolicy(c.observation_space, c.action_space, lambda _: 0., **policy_kwargs(c)).to(c.device)
        avc_init(policy, c)
        step = 0
    return PolicyAdapter(policy), step
