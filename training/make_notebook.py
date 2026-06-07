"""Generate a Colab .ipynb that bundles engine, bot, encoder, network, and training."""
import json
from pathlib import Path

ROOT = Path(__file__).parent

def code_cell(source: str):
    lines = source.splitlines(keepends=True)
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': lines,
    }

def md_cell(source: str):
    lines = source.splitlines(keepends=True)
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': lines,
    }


def read(name: str) -> str:
    return (ROOT / name).read_text()


def strip_main(src: str) -> str:
    """Drop the `if __name__ == '__main__':` block — its imports break in a notebook
    where everything is already in scope, and the smoke tests are unnecessary."""
    marker = "if __name__ == '__main__':"
    i = src.find(marker)
    if i == -1: return src
    return src[:i].rstrip() + '\n'


cells = []

# Header
cells.append(md_cell("""# Least Count — Trained Bot

Self-contained notebook to train a neural-net policy that plays Least Count better than the
heuristic Smart bot. Run cells top to bottom.

**Runtime:** Python 3, T4 GPU.

**Pipeline:**
1. Setup + GPU check.
2. Engine + tests (must all pass).
3. Smart bot + encoder + network.
4. Behavior cloning warm-start (~5 min).
5. PPO self-play training with league play (~6–10 hours).
6. Evaluation vs Smart.
7. Export weights as JSON for the browser.

Checkpoints save every 30 min; the best-vs-Smart model is saved separately.
"""))

# Cell 1: Setup
cells.append(md_cell("## Cell 1: Setup + GPU check"))
cells.append(code_cell("""\
import torch, numpy, sys
print('python', sys.version)
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device:', device)
"""))

# Cell 2: Engine
cells.append(md_cell("## Cell 2: Engine (Python port of the JS Game class)\n\nBit-for-bit equivalent to `index.html`."))
cells.append(code_cell(read('engine.py')))

# Cell 3: Engine tests
cells.append(md_cell("## Cell 3: Engine tests (must all pass)\n\nIf this fails, STOP — engine bug would corrupt training."))
# Need to inline the tests but reference engine that's already in the namespace.
# The test file imports from engine; in the notebook, engine is already defined globally.
# Strip the `from engine import ...` line.
test_src = read('test_engine.py')
# Replace the engine import with a no-op (everything's in scope)
test_src = test_src.replace(
    'from engine import Game, make_deck, RANKS, SUITS, DECLARE_MAX, WAR_RANK, SKIP_RANK, WAR_PER_CARD',
    '# engine symbols already in scope'
)
# Strip the sys.exit lines — in a notebook, SystemExit displays as a (misleading) traceback.
test_src = test_src.replace(
    'import sys\nsys.exit(0 if fail_n == 0 else 1)',
    "assert fail_n == 0, f'{fail_n} test(s) failed — stop and fix before training'"
)
cells.append(code_cell(test_src))

# Cell 4: Smart bot
cells.append(md_cell("## Cell 4: Smart bot (baseline + behavior cloning teacher)"))
bot_src = read('bot_smart.py')
bot_src = bot_src.replace('from engine import Game', '# Game already in scope')
cells.append(code_cell(strip_main(bot_src)))

# Cell 5: Encoder
cells.append(md_cell("## Cell 5: State encoder + action space"))
enc_src = read('encoder.py')
enc_src = enc_src.replace('from engine import Game', '# Game already in scope')
# The encoder file has a __main__ block that imports from bot_smart; that's fine in a script
# but in a notebook, the imports are unnecessary since everything is in scope.
# We can leave it; it won't execute since __main__ won't be set.
cells.append(code_cell(strip_main(enc_src)))

# Cell 6: Network
cells.append(md_cell("## Cell 6: Policy + value network"))
net_src = read('network.py')
net_src = net_src.replace('from encoder import STATE_DIM, ACTION_DIM', '# STATE_DIM/ACTION_DIM in scope')
cells.append(code_cell(net_src))

# Cell 7: Behavior cloning
cells.append(md_cell("## Cell 7: Behavior cloning module"))
bc_src = read('behavior_clone.py')
for old in [
    'from engine import Game',
    'from bot_smart import bot_smart, apply_action',
    'from encoder import (\n    Observer, encode_state, legal_action_mask,\n    smart_action_to_action_idx, STATE_DIM, ACTION_DIM\n)',
    'from network import PolicyValueNet',
]:
    bc_src = bc_src.replace(old, '# imports in scope')
cells.append(code_cell(bc_src))

# Cell 8: PPO training module
cells.append(md_cell("## Cell 8: PPO training module"))
ppo_src = read('ppo_train.py')
for old in [
    'from engine import Game',
    'from bot_smart import bot_smart, apply_action',
    'from encoder import (\n    Observer, encode_state, legal_action_mask, action_to_move,\n    STATE_DIM, ACTION_DIM\n)',
    'from network import PolicyValueNet, masked_policy',
]:
    ppo_src = ppo_src.replace(old, '# imports in scope')
cells.append(code_cell(ppo_src))

# Cell 9: BC pre-training
cells.append(md_cell("""## Cell 9: Behavior cloning warm-start (~5–10 min)

Collect ~30K state/action pairs from Smart vs Smart games, then supervised-train the policy
to imitate Smart. After this, the policy plays roughly at Smart level, which is a much better
starting point for PPO than random."""))
cells.append(code_cell("""\
print('Collecting BC samples from Smart vs Smart games...')
states, actions, masks = collect_smart_samples(num_games=2000, seed=42)
print(f'Collected {len(states)} samples (state dim {states.shape[1]}).')

net = PolicyValueNet().to(device)
print(f'Network: {count_params(net):,} params')
print('Behavior cloning...')
train_bc(net, states, actions, masks, epochs=8, batch_size=512, lr=1e-3, device=device)

# Save BC checkpoint
torch.save(net.state_dict(), '/content/policy_bc.pt')
print('Saved /content/policy_bc.pt')

# Quick eval vs Smart
print('Evaluating BC policy vs Smart...')
wr = eval_vs_smart(net, device, num_games=30, target=50)
print(f'BC win rate vs Smart at target=50: {wr*100:.0f}%')
"""))

# Cell 9b: Resume from checkpoint
cells.append(md_cell("""## (Optional) Resume from a saved checkpoint

If Colab disconnected mid-training, run this cell instead of Cell 9 (BC) to pick up from the
last checkpoint. Then continue with Cell 10."""))
cells.append(code_cell("""\
import os
# Try latest checkpoint first, then best
ckpt = None
for path in ['/content/policy_latest.pt', '/content/policy_best.pt', '/content/policy_bc.pt']:
    if os.path.exists(path):
        ckpt = path
        break

if ckpt is None:
    print('No checkpoint found. Run Cell 9 (BC) first.')
else:
    net = PolicyValueNet().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.train()
    print(f'Resumed from {ckpt}')
    # Quick eval to confirm
    wr = eval_vs_smart(net, device, num_games=20, target=50)
    print(f'Current win rate vs Smart: {wr*100:.0f}%')
"""))

# Cell 10: PPO training (improved — value pretrain, lower LR, more Smart in league, anti-collapse)
cells.append(md_cell("""## Cell 10: PPO self-play with league + value pretrain + anti-collapse

Improved over the first run. Key changes:
- **Value pretrain** before PPO: critic learns Monte Carlo returns from Smart games (prevents bad-value→bad-gradient cascade).
- **LR**: 1e-4 (was 3e-4) — slower, more stable updates.
- **Clip ratio**: 0.1 (was 0.2) — smaller per-step policy changes.
- **40% Smart in league** (was 20%) — anchors policy.
- **Linear LR warmup** for first 10K games.
- **Reward shaping**: small per-step reward for hand-pt reduction (denser signal).
- **Anti-collapse**: if win rate drops 10pp below best for 3 evals, roll back to best + halve LR.

Run AFTER Cell 9 (BC). ~4-8 hours."""))
cells.append(code_cell("""\
# Hyperparams
TOTAL_GAMES = 200_000
EVAL_EVERY = 5_000
CKPT_EVERY_MIN = 30
BATCH_SIZE = 4096
LR_BASE = 1e-4              # Was 3e-4 (lowered for stability)
LR_WARMUP_GAMES = 10_000
CLIP_RATIO = 0.1            # Was 0.2 (tighter)
TARGET = 50
SMART_FRAC = 0.40           # Was 0.20

# Step 1: Pretrain value head on Smart-vs-Smart MC returns
print('Pretraining value head on Smart returns...')
pretrain_value(net, device, num_games=600, epochs=3, batch_size=256, lr=5e-4)

# Eval after value pretrain
net.eval()
wr = eval_vs_smart(net, device, num_games=30, target=TARGET)
print(f'After BC + value pretrain: win rate vs Smart = {wr*100:.0f}%')

# Step 2: PPO
net.train()
optimizer = optim.Adam(net.parameters(), lr=LR_BASE)

import copy
def snapshot():
    snap = PolicyValueNet().to(device)
    snap.load_state_dict(copy.deepcopy(net.state_dict()))
    snap.eval()
    return snap

league = [snapshot()]
best_winrate = wr  # start from post-BC+value baseline
torch.save(net.state_dict(), '/content/policy_best.pt')
last_ckpt_t = time.time()
games_played = 0
batch = []
rng = random.Random(0)
regression_count = 0  # consecutive evals below best

start_t = time.time()
print(f'Starting PPO at {time.strftime("%H:%M:%S")}')

while games_played < TOTAL_GAMES:
    # LR warmup
    if games_played < LR_WARMUP_GAMES:
        warmup_lr = LR_BASE * (0.1 + 0.9 * games_played / LR_WARMUP_GAMES)
        for g_ in optimizer.param_groups:
            g_['lr'] = warmup_lr

    opp_kind, opponent = pick_opponent(league, device,
                                       lambda g, p: bot_smart(g, p),
                                       rng, smart_frac=SMART_FRAC)
    traj, winner = play_self_game(net, opponent, device, target=TARGET,
                                  rng=random.Random(), shape_reward=True)
    games_played += 1

    advs, rets = compute_gae(traj)
    for k, t in enumerate(traj):
        batch.append(t + (advs[k], rets[k]))

    if len(batch) >= BATCH_SIZE:
        ppo_update(net, optimizer, batch, device, clip_ratio=CLIP_RATIO)
        batch = []

    if games_played % EVAL_EVERY == 0:
        net.eval()
        wr = eval_vs_smart(net, device, num_games=40, target=TARGET)
        elapsed_min = (time.time() - start_t) / 60
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'[{games_played:>6}/{TOTAL_GAMES} games, {elapsed_min:.0f} min, lr={cur_lr:.1e}] win vs Smart: {wr*100:.0f}%')
        if wr > best_winrate:
            best_winrate = wr
            regression_count = 0
            torch.save(net.state_dict(), '/content/policy_best.pt')
            print(f'  ↑ new best ({wr*100:.0f}%), saved /content/policy_best.pt')
        elif wr < best_winrate - 0.10:
            regression_count += 1
            print(f'  ↓ regression ({wr*100:.0f}% vs best {best_winrate*100:.0f}%); count={regression_count}/3')
            if regression_count >= 3:
                # Roll back to best and lower LR
                net.load_state_dict(torch.load('/content/policy_best.pt'))
                new_lr = max(optimizer.param_groups[0]['lr'] * 0.5, 1e-5)
                for g_ in optimizer.param_groups:
                    g_['lr'] = new_lr
                regression_count = 0
                print(f'  ↩ rolled back to best, LR -> {new_lr:.1e}')
        else:
            regression_count = max(0, regression_count - 1)
        net.train()
        if games_played % (EVAL_EVERY * 2) == 0 and len(league) < 10:
            league.append(snapshot())

    if time.time() - last_ckpt_t > CKPT_EVERY_MIN * 60:
        torch.save(net.state_dict(), '/content/policy_latest.pt')
        last_ckpt_t = time.time()

torch.save(net.state_dict(), '/content/policy_final.pt')
print(f'Done. Best win rate vs Smart: {best_winrate*100:.0f}%')
"""))

# Cell 11: Final evaluation
cells.append(md_cell("""## Cell 11: Final evaluation

Loads the best checkpoint, plays 100 matches vs Smart at target=100, reports win rate."""))
cells.append(code_cell("""\
# Load the best checkpoint
best_net = PolicyValueNet().to(device)
best_net.load_state_dict(torch.load('/content/policy_best.pt', map_location=device))
best_net.eval()
print('Loaded /content/policy_best.pt')

# Evaluate at target=50
wr50 = eval_vs_smart(best_net, device, num_games=100, target=50)
print(f'Win rate vs Smart at target=50: {wr50*100:.0f}% ({int(wr50*100)} of 100)')

# Evaluate at target=100
wr100 = eval_vs_smart(best_net, device, num_games=50, target=100)
print(f'Win rate vs Smart at target=100: {wr100*100:.0f}% ({int(wr100*50)} of 50)')
"""))

# Cell 12: Export weights
cells.append(md_cell("""## Cell 12: Export weights for the browser

Saves the trained network's weights as a JSON file. Download it and send the path back to me;
I'll integrate it into `index.html` with a pure-JS inference function."""))
cells.append(code_cell("""\
import json
weights = {}
for name, param in best_net.state_dict().items():
    weights[name] = param.detach().cpu().numpy().tolist()

with open('/content/policy_weights.json', 'w') as f:
    json.dump({
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'hidden_dim': 256,
        'weights': weights,
        'training_info': {
            'win_rate_vs_smart_t50': wr50,
            'win_rate_vs_smart_t100': wr100,
        }
    }, f)

import os
size_kb = os.path.getsize('/content/policy_weights.json') / 1024
print(f'Saved /content/policy_weights.json ({size_kb:.0f} KB)')

# Trigger download in Colab
from google.colab import files
files.download('/content/policy_weights.json')
"""))

# Save notebook
notebook = {
    'cells': cells,
    'metadata': {
        'colab': {'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU',
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}

out_path = ROOT.parent / 'LeastCount_Training.ipynb'
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)

import os
size_kb = os.path.getsize(out_path) / 1024
print(f'Wrote {out_path} ({size_kb:.0f} KB, {len(cells)} cells)')
