"""
Compare the OLD training settings vs the NEW (post-fix) settings on the same machine.
Each run: BC + (value pretrain in new) + N PPO games, eval at intervals.

If NEW is better than OLD at the same number of games, the fixes work.
"""
import random
import time
import copy
import torch
import torch.optim as optim
import numpy as np

from engine import Game
from bot_smart import bot_smart, apply_action
from encoder import Observer, encode_state, legal_action_mask, action_to_move, STATE_DIM
from network import PolicyValueNet
from behavior_clone import collect_smart_samples, train_bc
from ppo_train import (
    pick_opponent, play_self_game, compute_gae, ppo_update,
    eval_vs_smart, pretrain_value
)

device = torch.device('cpu')
N_PPO_GAMES = 1500          # small enough for CPU in reasonable time
EVAL_EVERY = 250
BATCH_SIZE = 1024            # smaller for faster updates
EVAL_NUM_GAMES = 30


def snapshot(net):
    snap = PolicyValueNet().to(device)
    snap.load_state_dict(copy.deepcopy(net.state_dict()))
    snap.eval()
    return snap


def make_bc_net(states, actions, masks, label):
    """Fresh BC-trained net for fair comparison."""
    net = PolicyValueNet().to(device)
    print(f'  [{label}] BC training...')
    train_bc(net, states, actions, masks, epochs=4, batch_size=256, lr=1e-3, device=device)
    return net


def run_training(label, net, *,
                 lr=3e-4, clip_ratio=0.2, smart_frac=0.20,
                 shape_reward=False, use_value_pretrain=False,
                 anti_collapse=False, lr_warmup=False):
    """Run N_PPO_GAMES of PPO with the specified settings. Return list of (games, win_rate)."""
    print(f'\n=== {label} ===')
    print(f'  lr={lr}, clip={clip_ratio}, smart_frac={smart_frac}, shape_reward={shape_reward}, '
          f'value_pretrain={use_value_pretrain}, anti_collapse={anti_collapse}, lr_warmup={lr_warmup}')

    if use_value_pretrain:
        print(f'  [{label}] Value pretrain...')
        pretrain_value(net, device, num_games=200, epochs=2, batch_size=256, lr=5e-4)

    # Baseline eval
    net.eval()
    wr0 = eval_vs_smart(net, device, num_games=EVAL_NUM_GAMES, target=30)
    print(f'  [{label}] start win rate vs Smart (T=30): {wr0*100:.0f}%')
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    history = [(0, wr0)]
    league = [snapshot(net)]
    best_winrate = wr0
    best_state = copy.deepcopy(net.state_dict())
    regression_count = 0
    batch = []
    rng = random.Random(0)

    t0 = time.time()
    for gi in range(1, N_PPO_GAMES + 1):
        if lr_warmup and gi < 200:
            warm_lr = lr * (0.1 + 0.9 * gi / 200)
            for g_ in optimizer.param_groups:
                g_['lr'] = warm_lr

        opp_kind, opponent = pick_opponent(league, device,
                                           lambda g, p: bot_smart(g, p),
                                           rng, smart_frac=smart_frac)
        traj, winner = play_self_game(net, opponent, device, target=30,
                                      rng=random.Random(), shape_reward=shape_reward)
        advs, rets = compute_gae(traj)
        for k, t in enumerate(traj):
            batch.append(t + (advs[k], rets[k]))
        if len(batch) >= BATCH_SIZE:
            ppo_update(net, optimizer, batch, device, clip_ratio=clip_ratio, minibatch=128)
            batch = []

        if gi % EVAL_EVERY == 0:
            net.eval()
            wr = eval_vs_smart(net, device, num_games=EVAL_NUM_GAMES, target=30)
            elapsed = time.time() - t0
            print(f'  [{label}] {gi:>5} games ({elapsed:.0f}s) — win rate: {wr*100:.0f}%')
            history.append((gi, wr))
            if wr > best_winrate:
                best_winrate = wr
                best_state = copy.deepcopy(net.state_dict())
                regression_count = 0
            elif anti_collapse and wr < best_winrate - 0.10:
                regression_count += 1
                if regression_count >= 3:
                    print(f'  [{label}] rollback (regressed {(best_winrate-wr)*100:.0f}pp 3x); LR /= 2')
                    net.load_state_dict(best_state)
                    new_lr = max(optimizer.param_groups[0]['lr'] * 0.5, 1e-5)
                    for g_ in optimizer.param_groups:
                        g_['lr'] = new_lr
                    regression_count = 0
            else:
                regression_count = max(0, regression_count - 1)
            net.train()
            if gi % (EVAL_EVERY * 3) == 0 and len(league) < 8:
                league.append(snapshot(net))

    return history, best_winrate


# --- collect BC samples ONCE, reuse for both runs ---
print('Collecting BC samples (300 Smart vs Smart games)...')
t0 = time.time()
states, actions, masks = collect_smart_samples(num_games=300, seed=42)
print(f'  {len(states)} samples in {time.time()-t0:.0f}s')

# --- Run OLD settings ---
net_old = make_bc_net(states, actions, masks, 'OLD')
hist_old, best_old = run_training(
    'OLD', net_old,
    lr=3e-4, clip_ratio=0.2, smart_frac=0.20,
    shape_reward=False, use_value_pretrain=False,
    anti_collapse=False, lr_warmup=False,
)

# --- Run NEW settings ---
net_new = make_bc_net(states, actions, masks, 'NEW')
hist_new, best_new = run_training(
    'NEW', net_new,
    lr=1e-4, clip_ratio=0.1, smart_frac=0.40,
    shape_reward=True, use_value_pretrain=True,
    anti_collapse=True, lr_warmup=True,
)

# --- Compare ---
print('\n\n========== COMPARISON ==========')
print(f'{"games":>8}  {"OLD":>6}  {"NEW":>6}')
for (g, wo), (_, wn) in zip(hist_old, hist_new):
    diff = (wn - wo) * 100
    arrow = '↑' if diff > 0 else ('↓' if diff < -0.5 else '=')
    print(f'{g:>8}  {wo*100:>5.0f}%  {wn*100:>5.0f}%   {arrow} {diff:+.0f}pp')

print(f'\nOLD best: {best_old*100:.0f}%')
print(f'NEW best: {best_new*100:.0f}%')
print(f'NEW − OLD best: {(best_new-best_old)*100:+.0f} pp')
