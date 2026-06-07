"""
Behavior cloning: collect (state, smart_action) pairs from Smart vs Smart games,
then supervised pre-train the policy to imitate Smart. This is the warm-start for PPO.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from engine import Game
from bot_smart import bot_smart, apply_action
from encoder import (
    Observer, encode_state, legal_action_mask,
    smart_action_to_action_idx, STATE_DIM, ACTION_DIM
)
from network import PolicyValueNet


def collect_smart_samples(num_games: int = 1000, seed: int = 0):
    """Run Smart vs Smart games, record (state, action_idx, mask) tuples."""
    states, actions, masks = [], [], []
    rng = random.Random(seed)
    for gi in range(num_games):
        g = Game(target=100, names=['A','B'], rng=random.Random(seed + gi))
        obs = [Observer(0), Observer(1)]
        steps = 0
        while g.phase != 'gameover' and steps < 1500:
            if g.phase == 'roundover':
                g.next_round()
                continue
            p = g.turn
            obs[p].maybe_reset(g)
            state = encode_state(g, p, obs[p])
            mask = legal_action_mask(g, p)
            try:
                smart_a = bot_smart(g, p)
                aidx = smart_action_to_action_idx(g, p, smart_a)
                if not mask[aidx]:
                    steps += 1
                    apply_action(g, p, smart_a)
                    continue
                states.append(state)
                actions.append(aidx)
                masks.append(mask)
                # record opp throw for observer (other player's view)
                opp = p ^ 1
                obs[opp].record_opp_action(p, smart_a, list(g.floor))
                apply_action(g, p, smart_a)
            except Exception:
                break
            steps += 1
    return np.array(states), np.array(actions), np.array(masks)


def train_bc(net: PolicyValueNet, states, actions, masks,
             epochs: int = 5, batch_size: int = 256, lr: float = 1e-3,
             device='cpu'):
    """Supervised cross-entropy on Smart's action choices."""
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).long().to(device)
    masks_t = torch.from_numpy(masks).bool().to(device)
    N = len(states_t)
    print(f'  BC training: {N} samples, {epochs} epochs, batch={batch_size}')
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0
        correct = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            s = states_t[idx]
            a = actions_t[idx]
            m = masks_t[idx]
            logits, _ = net(s)
            # Mask illegal actions in CE: set their logit to large negative
            logits = logits.masked_fill(~m, -1e9)
            loss = nn.functional.cross_entropy(logits, a)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(idx)
            preds = logits.argmax(dim=-1)
            correct += (preds == a).sum().item()
        avg_loss = total_loss / N
        acc = correct / N
        print(f'  epoch {epoch+1}: loss={avg_loss:.4f} acc={acc*100:.1f}%')
    return net
