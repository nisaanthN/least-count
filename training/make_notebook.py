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
cells.append(code_cell(test_src))

# Cell 4: Smart bot
cells.append(md_cell("## Cell 4: Smart bot (baseline + behavior cloning teacher)"))
bot_src = read('bot_smart.py')
bot_src = bot_src.replace('from engine import Game', '# Game already in scope')
cells.append(code_cell(bot_src))

# Cell 5: Encoder
cells.append(md_cell("## Cell 5: State encoder + action space"))
enc_src = read('encoder.py')
enc_src = enc_src.replace('from engine import Game', '# Game already in scope')
# The encoder file has a __main__ block that imports from bot_smart; that's fine in a script
# but in a notebook, the imports are unnecessary since everything is in scope.
# We can leave it; it won't execute since __main__ won't be set.
cells.append(code_cell(enc_src))

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

# Cell 10: PPO training
cells.append(md_cell("""## Cell 10: PPO self-play with league (LONG — 4-10 hours)

Trains policy via self-play + league play. Evaluates vs Smart periodically. Auto-stops when win rate plateaus.

You can stop and resume; checkpoints save every 30 min."""))
cells.append(code_cell("""\
# Hyperparams
TOTAL_GAMES = 200_000        # Total self-play games
EVAL_EVERY = 5_000           # Eval vs Smart every N games
CKPT_EVERY_MIN = 30          # Save checkpoint every 30 min wall time
BATCH_SIZE = 4096            # Transitions per PPO update
LR = 3e-4
TARGET = 50                  # Train at target=50 for shorter games

# Initialize from BC
net.train()
optimizer = optim.Adam(net.parameters(), lr=LR)

# League: starts with current net only; we add snapshots as we go
import copy
def snapshot():
    snap = PolicyValueNet().to(device)
    snap.load_state_dict(copy.deepcopy(net.state_dict()))
    snap.eval()
    return snap

league = [snapshot()]
best_winrate = 0.0
last_ckpt_t = time.time()
games_played = 0
batch = []
rng = random.Random(0)

start_t = time.time()
print(f'Starting PPO at {time.strftime("%H:%M:%S")}')

while games_played < TOTAL_GAMES:
    # Sample opponent
    opp_kind, opponent = pick_opponent(league, device,
                                       lambda g, p: bot_smart(g, p),
                                       rng)
    # Play one game (net is player 0)
    traj, winner = play_self_game(net, opponent, device, target=TARGET, rng=random.Random())
    games_played += 1

    # Compute GAE + add to batch
    advs, rets = compute_gae(traj)
    for k, t in enumerate(traj):
        # t = (state, mask, aidx, lp, value, reward); extend with adv, ret
        batch.append(t + (advs[k], rets[k]))

    # Once batch is big enough, do PPO update
    if len(batch) >= BATCH_SIZE:
        ppo_update(net, optimizer, batch, device)
        batch = []

    # Periodic eval
    if games_played % EVAL_EVERY == 0:
        net.eval()
        wr = eval_vs_smart(net, device, num_games=40, target=TARGET)
        elapsed_min = (time.time() - start_t) / 60
        print(f'[{games_played:>6}/{TOTAL_GAMES} games, {elapsed_min:.0f} min] win rate vs Smart: {wr*100:.0f}%')
        if wr > best_winrate:
            best_winrate = wr
            torch.save(net.state_dict(), '/content/policy_best.pt')
            print(f'  ↑ new best ({wr*100:.0f}%), saved /content/policy_best.pt')
        net.train()
        # Add snapshot to league periodically
        if games_played % (EVAL_EVERY * 2) == 0 and len(league) < 10:
            league.append(snapshot())

    # Periodic checkpoint
    if time.time() - last_ckpt_t > CKPT_EVERY_MIN * 60:
        torch.save(net.state_dict(), '/content/policy_latest.pt')
        last_ckpt_t = time.time()

# Final save
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
