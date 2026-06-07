"""
Python port of botSmart and botEasy. Used as:
- Baseline opponent
- Behavior-cloning teacher
- League opponent during PPO training
"""
import random
from typing import List, Dict, Any, Optional
from engine import Game


def generate_throws(hand: List[Dict]) -> List[List[Dict]]:
    """Each single card + every same-rank set of size 2..n."""
    moves = []
    for c in hand:
        moves.append([c])
    by_rank: Dict[str, List[Dict]] = {}
    for c in hand:
        by_rank.setdefault(c['rank'], []).append(c)
    for r, cs in by_rank.items():
        if len(cs) >= 2:
            for k in range(2, len(cs) + 1):
                moves.append(cs[:k])
    return moves


def bot_easy(g: Game, p: int, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Random valid move; sometimes declares foolishly."""
    rng = rng or random
    view = g.view_for(p)
    if view['phase'] == 'war':
        hand = g.hands[p]
        non_sevens = [c for c in hand if c['rank'] != '7']
        if non_sevens:
            return {'t': 'discard', 'cardIds': [rng.choice(non_sevens)['id']]}
        return {'t': 'discard', 'cardIds': [hand[0]['id']]}
    if view['phase'] == 'discard':
        if g.hand_points(g.hands[p]) <= 5 and rng.random() < 0.35:
            return {'t': 'declare'}
        moves = generate_throws(g.hands[p])
        m = rng.choice(moves)
        return {'t': 'discard', 'cardIds': [c['id'] for c in m]}
    if view['phase'] == 'draw':
        pickable = [c for c in view['floor'] if c['rank'] not in ('7','J')]
        if pickable and rng.random() < 0.4:
            return {'t': 'draw', 'source': 'floor', 'cardId': rng.choice(pickable)['id']}
        return {'t': 'draw', 'source': 'deck'}
    return {'t': 'draw', 'source': 'deck'}


def score_throw_smart(g: Game, p: int, cards: List[Dict], view: Dict) -> float:
    hand_before = g.hands[p]
    chosen_ids = {c['id'] for c in cards}
    hand_after = [c for c in hand_before if c['id'] not in chosen_ids]
    pts_after = sum(g.card_points(c) for c in hand_after)
    pts_removed = sum(g.card_points(c) for c in cards)
    is_bulk = len(cards) >= 3
    is_seven = cards[0]['rank'] == '7'
    is_jack = cards[0]['rank'] == 'J'
    matches_pile = (len(view['floor']) > 0 and view['floor'][0]['rank'] == cards[0]['rank'])
    skips_draw = is_bulk or matches_pile
    score = -pts_after + pts_removed * 0.4
    if skips_draw: score += 8
    if len(hand_after) == 0: score += 120
    elif pts_after <= 5: score += 30
    if is_seven:
        if view['oppHandCount'] <= 4: score -= 10
        else: score += 4
        if len(hand_after) == 0: score += 20
    if is_jack:
        if pts_after <= 8: score += 14
        elif pts_after <= 15: score += 2
        else: score -= 4
        if len(cards) >= 3 and len(cards) % 2 == 1: score += 4
    if len(cards) == 1 and g.card_points(cards[0]) == 0: score -= 6
    if len(cards) == 1 and cards[0]['rank'] == 'A': score -= 3
    return score


def should_declare_smart(g: Game, p: int, view: Dict) -> bool:
    pts = g.hand_points(g.hands[p])
    if pts > 5: return False
    if view['oppHandCount'] >= 3: return True
    return pts == 0


def smart_draw(g: Game, p: int, view: Dict) -> Dict[str, Any]:
    for c in view['floor']:
        if c['rank'] in ('7','J'): continue
        if g.card_points(c) == 0:
            return {'t': 'draw', 'source': 'floor', 'cardId': c['id']}
        if any(x['rank'] == c['rank'] for x in g.hands[p]):
            return {'t': 'draw', 'source': 'floor', 'cardId': c['id']}
    return {'t': 'draw', 'source': 'deck'}


def smart_war(g: Game, p: int) -> Dict[str, Any]:
    hand = g.hands[p]
    sevens = [c for c in hand if c['rank'] == '7']
    if sevens:
        return {'t': 'discard', 'cardIds': [c['id'] for c in sevens]}
    moves = [m for m in generate_throws(hand) if m[0]['rank'] != '7']
    best = None
    best_s = float('-inf')
    for m in moves:
        pts = sum(g.card_points(c) for c in m)
        s = pts + len(m) * 1.5
        if s > best_s:
            best_s = s
            best = m
    return {'t': 'discard', 'cardIds': [c['id'] for c in best]}


def bot_smart(g: Game, p: int) -> Dict[str, Any]:
    view = g.view_for(p)
    if view['phase'] == 'war':
        return smart_war(g, p)
    if view['phase'] == 'discard':
        if should_declare_smart(g, p, view):
            return {'t': 'declare'}
        moves = generate_throws(g.hands[p])
        best = None
        best_s = float('-inf')
        for m in moves:
            s = score_throw_smart(g, p, m, view)
            if s > best_s:
                best_s = s
                best = m
        return {'t': 'discard', 'cardIds': [c['id'] for c in best]}
    if view['phase'] == 'draw':
        return smart_draw(g, p, view)
    return {'t': 'draw', 'source': 'deck'}


def apply_action(g: Game, p: int, a: Dict[str, Any]) -> None:
    t = a['t']
    if t == 'discard': g.discard(p, a['cardIds'])
    elif t == 'draw': g.draw(p, a['source'], a.get('cardId'))
    elif t == 'declare': g.declare(p)


def play_game(bot_a, bot_b, target: int = 50, seed: Optional[int] = None,
              max_steps: int = 5000) -> Optional[int]:
    """Play a full match. bot_a is player 0, bot_b is player 1. Returns winner index or None."""
    rng = random.Random(seed) if seed is not None else None
    g = Game(target=target, names=['A','B'], rng=rng)
    steps = 0
    while g.phase != 'gameover' and steps < max_steps:
        p = g.turn
        try:
            a = bot_a(g, p) if p == 0 else bot_b(g, p)
            apply_action(g, p, a)
        except Exception:
            break
        if g.phase == 'roundover':
            g.next_round()
        steps += 1
    return g.winner


if __name__ == '__main__':
    # Quick sanity: Smart vs Easy
    wins = [0, 0]
    for s in range(30):
        rng_a = random.Random(s * 2)
        rng_b = random.Random(s * 2 + 1)
        w = play_game(lambda g, p: bot_smart(g, p),
                      lambda g, p: bot_easy(g, p, rng_b),
                      target=50, seed=s)
        if w is not None:
            wins[w] += 1
    print(f'Smart vs Easy: {wins[0]}-{wins[1]} (Smart win rate {wins[0]/sum(wins)*100:.0f}%)')
