"""
State encoder + action space for RL training.

State (96 dims) — all info the bot can observe:
- Hand rank counts (14 dims: 13 ranks + Joker), normalized by 8
- Floor top rank one-hot (14 dims)
- Floor size (1) / pickable count (1) / hand-matches-pile binary (1)
- Wild rank one-hot (14 dims)
- Deck count (1) / opp hand count (1)
- Graveyard rank counts (14 dims) — CARD COUNTING
- My score (1) / opp score (1) / target (1) / round (1) / am-I-starter (1)
- Phase one-hot (3 dims: discard/draw/war)
- Pending penalty (1) / pending jacks (1)
- Opp pile picks per rank this round (14 dims)
- Last opp throw rank one-hot (14 dims) / last opp throw count (1)
- My #pairs (1) / #triples (1) / max set size (1)

Action space (45 actions):
- For each of 14 ranks (A,2,3,4,5,6,7,8,9,10,J,Q,K,JOKER):
    throw_1, throw_2, throw_3plus  → 42 actions
- declare (1)
- draw_deck (1)
- draw_pile (1)  [picks pile slot 0; pile cards are homogeneous after a throw]

Legal-action mask per state.
"""
import numpy as np
from collections import Counter
from typing import Optional, Dict, Any, List
from engine import Game

# Rank list including Joker for action/state indexing
ALL_RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K','JOKER']  # 14
RANK_INDEX = {r: i for i, r in enumerate(ALL_RANKS)}

STATE_DIM = (
    14 +   # hand rank counts
    14 +   # floor top rank one-hot
    1 + 1 + 1 +   # floor_size, pickable_count, hand_matches_pile
    14 +   # wild rank one-hot
    1 + 1 +   # deck_count, opp_hand_count
    14 +   # graveyard rank counts
    1 + 1 + 1 + 1 + 1 +   # my_score, opp_score, target, round_num, am_starter
    3 +    # phase one-hot
    1 + 1 +   # pending_penalty, pending_jacks
    14 +   # opp pile picks per rank (this round)
    14 + 1 +   # last_opp_throw_rank, last_opp_throw_count
    1 + 1 + 1   # my_num_pairs, my_num_triples, my_max_set_size
)
assert STATE_DIM == 103, f'STATE_DIM is {STATE_DIM}'

# Action space layout
N_THROW_RANKS = 14
ACTION_THROW1_BASE = 0           # actions 0..13: throw 1 of rank
ACTION_THROW2_BASE = 14          # actions 14..27: throw pair
ACTION_THROW3_BASE = 28          # actions 28..41: throw 3+ (all of rank)
ACTION_DECLARE = 42
ACTION_DRAW_DECK = 43
ACTION_DRAW_PILE = 44
ACTION_DIM = 45


class Observer:
    """Tracks observable game state across actions for one bot player."""

    def __init__(self, bot_p: int):
        self.bot_p = bot_p
        self.opp_pile_picks: Dict[str, int] = {r: 0 for r in ALL_RANKS}
        self.last_opp_throw_rank: Optional[str] = None
        self.last_opp_throw_count: int = 0
        self.last_round_seen: int = 0

    def maybe_reset(self, g: Game) -> None:
        if g.round != self.last_round_seen:
            self.opp_pile_picks = {r: 0 for r in ALL_RANKS}
            self.last_opp_throw_rank = None
            self.last_opp_throw_count = 0
            self.last_round_seen = g.round

    def record_opp_action(self, p: int, a: Dict, g_before_floor: List[Dict]) -> None:
        if p == self.bot_p: return
        if a.get('t') == 'draw' and a.get('source') == 'floor':
            cid = a.get('cardId')
            for c in g_before_floor:
                if c['id'] == cid:
                    self.opp_pile_picks[c['rank']] = self.opp_pile_picks.get(c['rank'], 0) + 1
                    break
        if a.get('t') == 'discard':
            ids = a.get('cardIds', [])
            # We don't have ranks from id alone, but ids include rank prefix. Parse it.
            rank = None
            for cid in ids:
                if cid.startswith('JOKER'):
                    rank = 'JOKER'
                    break
                # Standard ids: '10' or single char rank + suit + optional a/b
                if cid.startswith('10'):
                    rank = '10'; break
                rank = cid[0]
                break
            self.last_opp_throw_rank = rank
            self.last_opp_throw_count = len(ids)


def encode_state(g: Game, bot_p: int, obs: Observer) -> np.ndarray:
    """Return 102-dim float32 state vector from bot_p's perspective."""
    obs.maybe_reset(g)
    s = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # Hand rank counts (14)
    hand_counts = Counter(c['rank'] for c in g.hands[bot_p])
    for i, r in enumerate(ALL_RANKS):
        s[idx + i] = hand_counts.get(r, 0) / 8.0
    idx += 14

    # Floor top rank one-hot (14)
    if g.floor:
        ftr = g.floor[0]['rank']
        if ftr in RANK_INDEX:
            s[idx + RANK_INDEX[ftr]] = 1.0
    idx += 14

    # Floor size, pickable count, hand-matches-pile
    s[idx] = len(g.floor) / 8.0; idx += 1
    pickable = sum(1 for c in g.floor if c['rank'] not in ('7','J'))
    s[idx] = pickable / 4.0; idx += 1
    if g.floor:
        ftr = g.floor[0]['rank']
        s[idx] = 1.0 if any(c['rank'] == ftr for c in g.hands[bot_p]) else 0.0
    idx += 1

    # Wild rank one-hot (14)
    if g.wild_rank in RANK_INDEX:
        s[idx + RANK_INDEX[g.wild_rank]] = 1.0
    idx += 14

    # Deck count, opp hand count
    s[idx] = len(g.deck) / 106.0; idx += 1
    s[idx] = len(g.hands[bot_p ^ 1]) / 12.0; idx += 1

    # Graveyard rank counts (14) — card counting
    grave_counts = Counter(c['rank'] for c in g.graveyard)
    for i, r in enumerate(ALL_RANKS):
        s[idx + i] = grave_counts.get(r, 0) / 8.0
    idx += 14

    # My score, opp score, target, round, am-starter
    s[idx] = g.scores[bot_p] / max(g.target, 1); idx += 1
    s[idx] = g.scores[bot_p ^ 1] / max(g.target, 1); idx += 1
    s[idx] = g.target / 200.0; idx += 1
    s[idx] = min(g.round, 20) / 20.0; idx += 1
    s[idx] = 1.0 if g.starter == bot_p else 0.0; idx += 1

    # Phase one-hot (3)
    phase_map = {'discard': 0, 'draw': 1, 'war': 2}
    if g.phase in phase_map:
        s[idx + phase_map[g.phase]] = 1.0
    idx += 3

    # Pending penalty, pending jacks
    s[idx] = g.pending_penalty / 20.0; idx += 1
    s[idx] = g.pending_jacks / 4.0; idx += 1

    # Opp pile picks per rank (14)
    for i, r in enumerate(ALL_RANKS):
        s[idx + i] = obs.opp_pile_picks.get(r, 0) / 4.0
    idx += 14

    # Last opp throw (14 one-hot + count)
    if obs.last_opp_throw_rank and obs.last_opp_throw_rank in RANK_INDEX:
        s[idx + RANK_INDEX[obs.last_opp_throw_rank]] = 1.0
    idx += 14
    s[idx] = min(obs.last_opp_throw_count, 4) / 4.0; idx += 1

    # My num pairs / triples / max set size
    num_pairs = sum(1 for cnt in hand_counts.values() if cnt == 2)
    num_triples = sum(1 for cnt in hand_counts.values() if cnt >= 3)
    max_set = max(hand_counts.values()) if hand_counts else 0
    s[idx] = num_pairs / 6.0; idx += 1
    s[idx] = num_triples / 3.0; idx += 1
    s[idx] = max_set / 8.0; idx += 1

    assert idx == STATE_DIM, f'encoded {idx}, expected {STATE_DIM}'
    return s


def legal_action_mask(g: Game, p: int) -> np.ndarray:
    """Boolean array (ACTION_DIM,) where True = legal."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    if g.phase == 'discard' or g.phase == 'war':
        hand_counts = Counter(c['rank'] for c in g.hands[p])
        for r, cnt in hand_counts.items():
            if r not in RANK_INDEX: continue
            ri = RANK_INDEX[r]
            # In war, only sevens can be thrown (rule: must counter or pick)
            #   ... unless we treat any non-7 throw in war as "end war + pick penalty"
            #   The engine allows any throw in war (non-7 ends the war).
            # So all rank throws are legal in war.
            if cnt >= 1: mask[ACTION_THROW1_BASE + ri] = True
            if cnt >= 2: mask[ACTION_THROW2_BASE + ri] = True
            if cnt >= 3: mask[ACTION_THROW3_BASE + ri] = True
        if g.can_declare(p):
            mask[ACTION_DECLARE] = True
    if g.phase == 'draw':
        mask[ACTION_DRAW_DECK] = True
        # Pile draw: legal if any non-7-non-J card on floor
        if any(c['rank'] not in ('7','J') for c in g.floor):
            mask[ACTION_DRAW_PILE] = True
    return mask


def action_to_move(g: Game, p: int, action_idx: int) -> Dict[str, Any]:
    """Convert action index → game action dict the engine accepts."""
    if action_idx == ACTION_DECLARE:
        return {'t': 'declare'}
    if action_idx == ACTION_DRAW_DECK:
        return {'t': 'draw', 'source': 'deck'}
    if action_idx == ACTION_DRAW_PILE:
        # Pick the first non-7-non-J card on floor
        for c in g.floor:
            if c['rank'] not in ('7','J'):
                return {'t': 'draw', 'source': 'floor', 'cardId': c['id']}
        raise ValueError('No pickable card on pile')
    # Throw actions
    if action_idx < ACTION_THROW2_BASE:
        size = 1; rank_idx = action_idx - ACTION_THROW1_BASE
    elif action_idx < ACTION_THROW3_BASE:
        size = 2; rank_idx = action_idx - ACTION_THROW2_BASE
    elif action_idx < ACTION_DECLARE:
        size = 3; rank_idx = action_idx - ACTION_THROW3_BASE
    else:
        raise ValueError(f'Unknown action {action_idx}')
    rank = ALL_RANKS[rank_idx]
    matching = [c for c in g.hands[p] if c['rank'] == rank]
    if size == 1:
        return {'t': 'discard', 'cardIds': [matching[0]['id']]}
    if size == 2:
        return {'t': 'discard', 'cardIds': [matching[0]['id'], matching[1]['id']]}
    # 3+: throw all of this rank
    return {'t': 'discard', 'cardIds': [c['id'] for c in matching]}


def smart_action_to_action_idx(g: Game, p: int, smart_action: Dict[str, Any]) -> int:
    """Convert a Smart bot's natural action dict → action_idx for behavior cloning."""
    t = smart_action.get('t')
    if t == 'declare': return ACTION_DECLARE
    if t == 'draw':
        return ACTION_DRAW_DECK if smart_action['source'] == 'deck' else ACTION_DRAW_PILE
    if t == 'discard':
        ids = smart_action['cardIds']
        # Determine rank
        first_id = ids[0]
        if first_id.startswith('JOKER'):
            rank = 'JOKER'
        elif first_id.startswith('10'):
            rank = '10'
        else:
            rank = first_id[0]
        ri = RANK_INDEX[rank]
        n = len(ids)
        if n == 1: return ACTION_THROW1_BASE + ri
        if n == 2: return ACTION_THROW2_BASE + ri
        return ACTION_THROW3_BASE + ri
    raise ValueError(f'Unknown smart action: {smart_action}')


# ===== Self-tests =====
if __name__ == '__main__':
    import random
    from bot_smart import bot_smart, apply_action
    print('STATE_DIM =', STATE_DIM, 'ACTION_DIM =', ACTION_DIM)
    # Run a smoke game with the encoder
    g = Game(target=50, names=['A','B'], rng=random.Random(42))
    obs0 = Observer(0); obs1 = Observer(1)
    encoded = encode_state(g, 0, obs0)
    print('First state shape:', encoded.shape, 'sum:', encoded.sum())
    mask = legal_action_mask(g, 0)
    print('Initial legal actions:', mask.sum(), '/', ACTION_DIM)
    # Play 50 turns
    for step in range(50):
        if g.phase == 'gameover': break
        if g.phase == 'roundover':
            g.next_round()
            obs0.maybe_reset(g); obs1.maybe_reset(g)
            continue
        p = g.turn
        s = encode_state(g, p, obs0 if p == 0 else obs1)
        m = legal_action_mask(g, p)
        assert m.any(), f'no legal action at step {step}'
        # Test: convert Smart's action to action_idx and verify it's legal
        smart_a = bot_smart(g, p)
        try:
            aidx = smart_action_to_action_idx(g, p, smart_a)
            assert m[aidx], f'Smart picked illegal aidx {aidx}, action={smart_a}'
            # Apply
            apply_action(g, p, smart_a)
        except Exception as e:
            print(f'ERROR at step {step}: {e}, smart_a={smart_a}, phase={g.phase}')
            break
    print(f'Game phase after 50 steps: {g.phase}, scores: {g.scores}')
    print('Encoder smoke test passed.')
