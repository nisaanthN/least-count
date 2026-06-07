"""End-to-end smoke test: tiny version of the full pipeline. Catches runtime bugs."""
import sys
import time
import random
import torch
import torch.optim as optim

from engine import Game
from bot_smart import bot_smart, apply_action
from encoder import (
    Observer, encode_state, legal_action_mask, action_to_move,
    smart_action_to_action_idx, STATE_DIM, ACTION_DIM
)
from network import PolicyValueNet, count_params
from behavior_clone import collect_smart_samples, train_bc
from ppo_train import (
    pick_opponent, policy_action, opp_action, play_self_game,
    compute_gae, ppo_update, eval_vs_smart
)

device = torch.device('cpu')
print(f'STATE_DIM={STATE_DIM} ACTION_DIM={ACTION_DIM}')

# Stage 1: BC sample collection
print('\n[1/5] Collecting BC samples (50 games)...')
t0 = time.time()
states, actions, masks = collect_smart_samples(num_games=50, seed=42)
print(f'  collected {len(states)} samples in {time.time()-t0:.1f}s')
assert states.shape[1] == STATE_DIM
assert masks.shape[1] == ACTION_DIM

# Stage 2: BC training
print('\n[2/5] Behavior cloning (2 epochs)...')
net = PolicyValueNet().to(device)
print(f'  network: {count_params(net):,} params')
train_bc(net, states, actions, masks, epochs=2, batch_size=256, lr=1e-3, device=device)

# Stage 3: Quick eval vs Smart
print('\n[3/5] Eval BC policy vs Smart (5 games)...')
wr = eval_vs_smart(net, device, num_games=5, target=30)
print(f'  win rate vs Smart at T=30: {wr*100:.0f}%')

# Stage 4: PPO self-play - 10 games
print('\n[4/5] PPO self-play (10 games against league)...')
net.train()
optimizer = optim.Adam(net.parameters(), lr=3e-4)
import copy
def snapshot():
    snap = PolicyValueNet().to(device)
    snap.load_state_dict(copy.deepcopy(net.state_dict()))
    snap.eval()
    return snap

league = [snapshot()]
rng = random.Random(0)
batch = []
for gi in range(10):
    opp_kind, opponent = pick_opponent(league, device,
                                       lambda g, p: bot_smart(g, p),
                                       rng)
    traj, winner = play_self_game(net, opponent, device, target=30,
                                  rng=random.Random(gi + 1000))
    print(f'  game {gi+1}/10: opp={opp_kind}, winner={winner}, traj_len={len(traj)}')
    advs, rets = compute_gae(traj)
    for k, t in enumerate(traj):
        batch.append(t + (advs[k], rets[k]))

print(f'  total batch: {len(batch)} transitions')

# Stage 5: Run a PPO update
print('\n[5/5] PPO update on collected batch...')
if len(batch) >= 32:
    ppo_update(net, optimizer, batch, device, minibatch=32)
    print('  PPO update completed')
else:
    print(f'  skipped (only {len(batch)} transitions)')

# Final sanity eval
print('\n[Final] Eval vs Smart (5 more games)...')
net.eval()
wr2 = eval_vs_smart(net, device, num_games=5, target=30)
print(f'  win rate vs Smart at T=30: {wr2*100:.0f}%')

print('\n✓ SMOKE TEST PASSED — pipeline runs end-to-end without errors.')
