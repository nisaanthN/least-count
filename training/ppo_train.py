"""
PPO self-play training with league play.
Single-process; uses one GPU for the policy net. Plays games sequentially using the Python engine.

Key features:
- Bot policy = same network for both seats in self-play; we sample actions and compute reward at round/match end.
- League: 50% latest, 30% earlier checkpoints, 20% Smart heuristic.
- GAE-based advantage with γ=0.99, λ=0.95.
- PPO clip ε=0.2, value coef 0.5, entropy coef 0.01.
- Checkpoint every 30 min wall time. Best-vs-Smart checkpoint separately.
"""
import random
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional
from engine import Game
from bot_smart import bot_smart, apply_action
from encoder import (
    Observer, encode_state, legal_action_mask, action_to_move,
    STATE_DIM, ACTION_DIM
)
from network import PolicyValueNet, masked_policy


def pick_opponent(league: List[PolicyValueNet], device, smart_fn,
                  rng: random.Random, smart_frac: float = 0.40):
    """Sample an opponent.
    smart_frac fraction of games are vs Smart (anchors policy to known-good baseline).
    Of the remaining: 60% latest checkpoint, 40% earlier checkpoints (when available)."""
    r = rng.random()
    if r < smart_frac or not league:
        return 'smart', smart_fn
    # Of the non-Smart portion, half-half-ish split between latest and past
    r2 = rng.random()
    if r2 < 0.40 and len(league) > 1:
        opp = rng.choice(league[:-1])
        return 'past', opp
    return 'latest', league[-1]


def policy_action(net: PolicyValueNet, state: np.ndarray, mask: np.ndarray,
                  device: torch.device, deterministic: bool = False):
    """Sample action from policy. Returns (action_idx, logprob, value)."""
    s = torch.from_numpy(state).float().unsqueeze(0).to(device)
    m = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = net(s)
    dist = masked_policy(logits, m)
    if deterministic:
        # Pick argmax legal
        masked_logits = logits.masked_fill(~m, float('-inf'))
        action_idx = masked_logits.argmax(dim=-1).item()
    else:
        action_idx = dist.sample().item()
    logprob = dist.log_prob(torch.tensor([action_idx], device=device)).item()
    return action_idx, logprob, value.item()


def opp_action(opponent, g, p, device):
    """Get action from any opponent type (Smart fn or PPO net).
    Note: nn.Module is callable too, so we must check isinstance FIRST."""
    if isinstance(opponent, nn.Module):
        obs = Observer(p)
        obs.maybe_reset(g)
        state = encode_state(g, p, obs)
        mask = legal_action_mask(g, p)
        if not mask.any():
            return None
        aidx, _, _ = policy_action(opponent, state, mask, device, deterministic=False)
        return action_to_move(g, p, aidx)
    # Plain Smart bot function (lambda or similar)
    return opponent(g, p)


def play_self_game(net, opponent, device, target=50, max_steps=2000,
                   rng: Optional[random.Random] = None,
                   bot_p: int = 0, gamma: float = 0.99,
                   shape_reward: bool = True):
    """Play one game: net plays as bot_p, opponent plays as 1-bot_p.
    Returns list of (state, mask, action_idx, logprob, value, reward) for the net's actions.

    Reward shaping (when shape_reward=True):
      - small per-step reward = +0.1 * (hand_pts_before - hand_pts_after) — incentivizes dumping
      - round end: -(my_round_delta - opp_round_delta) (capped at ±40 already by engine)
      - match end: ±20 for win/loss (smaller than before so signal is less dominated by tail)
    """
    g = Game(target=target, names=['A','B'], rng=rng)
    obs = Observer(bot_p)
    traj = []
    last_score_diff = 0
    steps = 0
    pts_before_action = g.hand_points(g.hands[bot_p])
    while g.phase != 'gameover' and steps < max_steps:
        if g.phase == 'roundover':
            diff = g.scores[bot_p] - g.scores[bot_p ^ 1]
            round_reward = -(diff - last_score_diff)
            last_score_diff = diff
            if traj:
                traj[-1] = (*traj[-1][:5], traj[-1][5] + round_reward)
            g.next_round()
            obs.maybe_reset(g)
            pts_before_action = g.hand_points(g.hands[bot_p])
            continue
        p = g.turn
        if p == bot_p:
            obs.maybe_reset(g)
            state = encode_state(g, p, obs)
            mask = legal_action_mask(g, p)
            if not mask.any(): break
            aidx, lp, val = policy_action(net, state, mask, device, deterministic=False)
            a = action_to_move(g, p, aidx)
            try:
                apply_action(g, p, a)
            except Exception:
                break
            # Per-step shaping: reward for hand-pts reduction
            step_reward = 0.0
            if shape_reward:
                pts_after = g.hand_points(g.hands[bot_p])
                step_reward = 0.1 * (pts_before_action - pts_after)
                pts_before_action = pts_after
            traj.append((state, mask, aidx, lp, val, step_reward))
        else:
            a = opp_action(opponent, g, p, device)
            if a is None: break
            opp_floor_snapshot = list(g.floor)
            try:
                obs.record_opp_action(p, a, opp_floor_snapshot)
                apply_action(g, p, a)
            except Exception:
                break
            if shape_reward:
                pts_before_action = g.hand_points(g.hands[bot_p])
        steps += 1
    if g.winner is not None:
        # Match reward (smaller; round rewards already cover most signal)
        match_reward = 20.0 if g.winner == bot_p else -20.0
        if traj:
            traj[-1] = (*traj[-1][:5], traj[-1][5] + match_reward)
    return traj, g.winner


def pretrain_value(net, device, num_games: int = 1000, epochs: int = 3,
                   batch_size: int = 256, lr: float = 5e-4):
    """Fit the value head to Monte Carlo returns from Smart-vs-Smart games.
    Runs AFTER BC. Stabilizes PPO by giving the critic a good initialization."""
    from behavior_clone import collect_smart_samples  # already in scope in notebook
    print(f'  Value pretrain: rolling out {num_games} Smart vs Smart games to collect returns...')
    # Replay Smart-vs-Smart games, compute MC returns at each (bot, state).
    samples = []
    rng = random.Random(99)
    for gi in range(num_games):
        g = Game(target=50, names=['A','B'], rng=random.Random(gi + 7777))
        per_player_traj = {0: [], 1: []}
        per_player_last_diff = {0: 0, 1: 0}
        per_player_obs = {0: Observer(0), 1: Observer(1)}
        steps = 0
        while g.phase != 'gameover' and steps < 1500:
            if g.phase == 'roundover':
                for p in (0, 1):
                    diff = g.scores[p] - g.scores[p ^ 1]
                    rr = -(diff - per_player_last_diff[p])
                    per_player_last_diff[p] = diff
                    if per_player_traj[p]:
                        s, _ = per_player_traj[p][-1]
                        per_player_traj[p][-1] = (s, per_player_traj[p][-1][1] + rr)
                g.next_round()
                continue
            p = g.turn
            per_player_obs[p].maybe_reset(g)
            state = encode_state(g, p, per_player_obs[p])
            per_player_traj[p].append((state, 0.0))
            try:
                a = bot_smart(g, p)
                per_player_obs[p ^ 1].record_opp_action(p, a, list(g.floor))
                apply_action(g, p, a)
            except Exception:
                break
            steps += 1
        if g.winner is not None:
            for p in (0, 1):
                mr = 20.0 if g.winner == p else -20.0
                if per_player_traj[p]:
                    s, r = per_player_traj[p][-1]
                    per_player_traj[p][-1] = (s, r + mr)
        # Compute discounted returns
        for p in (0, 1):
            traj = per_player_traj[p]
            G = 0.0
            for i in reversed(range(len(traj))):
                s, r = traj[i]
                G = r + 0.99 * G
                samples.append((s, G))
    print(f'  Collected {len(samples)} (state, return) pairs')
    if not samples: return
    import numpy as np
    states_arr = np.stack([s for s, _ in samples])
    returns_arr = np.array([G for _, G in samples], dtype=np.float32)
    states_t = torch.from_numpy(states_arr).float().to(device)
    returns_t = torch.from_numpy(returns_arr).float().to(device)
    # Train only value head + shared (light)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    N = len(states_t)
    for ep in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            _, v = net(states_t[idx])
            loss = ((v - returns_t[idx]) ** 2).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f'  value epoch {ep+1}: MSE={total_loss/N:.3f}')


def compute_gae(traj, gamma: float = 0.99, lam: float = 0.95):
    """Returns advantages and value targets (returns)."""
    n = len(traj)
    if n == 0: return [], []
    rewards = [t[5] for t in traj]
    values = [t[4] for t in traj]
    advantages = [0.0] * n
    gae = 0.0
    next_value = 0.0  # bootstrapping value past end
    for i in reversed(range(n)):
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae
        advantages[i] = gae
        next_value = values[i]
    returns = [adv + v for adv, v in zip(advantages, values)]
    return advantages, returns


def ppo_update(net, optimizer, batch, device,
               clip_ratio: float = 0.2, value_coef: float = 0.5,
               entropy_coef: float = 0.01, n_epochs: int = 4,
               minibatch: int = 256):
    """Run PPO updates on a collected batch."""
    states = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device)
    masks = torch.from_numpy(np.stack([b[1] for b in batch])).bool().to(device)
    actions = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
    old_logprobs = torch.tensor([b[3] for b in batch], dtype=torch.float, device=device)
    advantages = torch.tensor([b[6] for b in batch], dtype=torch.float, device=device)
    returns = torch.tensor([b[7] for b in batch], dtype=torch.float, device=device)
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    N = len(states)
    for _ in range(n_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, minibatch):
            idx = perm[i:i+minibatch]
            s = states[idx]; m = masks[idx]; a = actions[idx]
            old_lp = old_logprobs[idx]; adv = advantages[idx]; ret = returns[idx]
            logits, v = net(s)
            dist = masked_policy(logits, m)
            new_lp = dist.log_prob(a)
            entropy = dist.entropy().mean()
            ratio = (new_lp - old_lp).exp()
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            policy_loss = -torch.min(ratio * adv, clip_adv).mean()
            value_loss = ((v - ret) ** 2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()


def eval_vs_smart(net, device, num_games: int = 30, target: int = 100) -> float:
    """Returns win rate against Smart over num_games matches."""
    wins = 0
    for s in range(num_games):
        rng = random.Random(s * 7919 + 17)
        # Alternate sides each game for fairness
        net_player = s % 2
        g = Game(target=target, names=['Net','Smart'], rng=rng)
        obs = Observer(net_player)
        steps = 0
        while g.phase != 'gameover' and steps < 3000:
            if g.phase == 'roundover':
                g.next_round()
                obs.maybe_reset(g)
                continue
            p = g.turn
            try:
                if p == net_player:
                    obs.maybe_reset(g)
                    state = encode_state(g, p, obs)
                    mask = legal_action_mask(g, p)
                    if not mask.any(): break
                    aidx, _, _ = policy_action(net, state, mask, device, deterministic=True)
                    a = action_to_move(g, p, aidx)
                else:
                    a = bot_smart(g, p)
                    obs.record_opp_action(p, a, list(g.floor))
                apply_action(g, p, a)
            except Exception:
                break
            steps += 1
        if g.winner == net_player:
            wins += 1
    return wins / num_games
