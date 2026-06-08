"""Convert a PyTorch state_dict checkpoint into a small JSON the browser can fetch.

Usage:
    python export_weights.py path/to/policy_best.pt [path/to/output.json]

Produces a JSON with float32 weights as nested lists, matching the keys of
PolicyValueNet's state_dict. The browser inference code in index.html consumes this.
"""
import json
import sys
from pathlib import Path
import torch

from network import PolicyValueNet
from encoder import STATE_DIM, ACTION_DIM


def main():
    if len(sys.argv) < 2:
        print('usage: export_weights.py <policy_best.pt> [out.json]', file=sys.stderr)
        sys.exit(2)
    ckpt = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else ckpt.with_suffix('.json')

    net = PolicyValueNet()
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd)
    net.eval()

    weights = {}
    for k, v in net.state_dict().items():
        weights[k] = v.detach().cpu().numpy().astype('float32').tolist()

    payload = {
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'hidden_dim': 256,
        'weights': weights,
    }
    out.write_text(json.dumps(payload))
    size_kb = out.stat().st_size / 1024
    print(f'wrote {out} ({size_kb:.0f} KB, {len(weights)} tensors)')


if __name__ == '__main__':
    main()
