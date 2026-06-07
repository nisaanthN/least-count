"""
Core test suite for the Python engine port. Mirrors the critical scenarios from /tmp/lc_test.js.
"""
import random
from engine import Game, make_deck, RANKS, SUITS, DECLARE_MAX, WAR_RANK, SKIP_RANK, WAR_PER_CARD

pass_n = 0
fail_n = 0


def ok(name, cond, info=''):
    global pass_n, fail_n
    if cond:
        pass_n += 1
        print(f'  ok   {name}')
    else:
        fail_n += 1
        print(f'  FAIL {name} {info}')


def H(s):
    """Parse card string like '5H', '10C', 'JK1', 'JK2'."""
    if s == 'JK1': return {'id': 'JOKERa', 'rank': 'JOKER', 'suit': None}
    if s == 'JK2': return {'id': 'JOKERb', 'rank': 'JOKER', 'suit': None}
    # Optional trailing a/b for 2-deck disambiguation
    import re
    m = re.match(r'^(10|[A2-9JQK])([SHDC])(\d?)$', s)
    if not m: raise ValueError(f'bad card {s}')
    suf = m.group(3) or ''
    return {'id': m.group(1) + m.group(2) + suf, 'rank': m.group(1), 'suit': m.group(2)}


def make_game(target=100, seed=None):
    rng = random.Random(seed) if seed is not None else None
    return Game(target=target, names=['A','B'], rng=rng)


def reset(g, turn=0):
    g.turn = turn
    g.phase = 'discard'
    g.pending_penalty = 0
    g.pending_discard = None
    g.pending_jacks = 0
    g.pending_sevens = 0
    g.skip_eligible = False
    g.war_starter = None
    g.graveyard = []


def set_hands(g, h0, h1):
    g.hands[0] = [H(s) for s in h0]
    g.hands[1] = [H(s) for s in h1]


def set_floor(g, cards):
    g.floor = [H(s) for s in cards]


def set_deck(g, top):
    """top[0] is the next card to be popped."""
    g.deck = [H(s) for s in reversed(top)]


def set_wild(g, rank):
    g.wild_indicator = {'id': rank + 'S', 'rank': rank, 'suit': 'S'}
    g.wild_rank = rank


# ==================== TESTS ====================

print('=== ENGINE PORT TESTS ===\n')

print('--- 1: makeDeck has 106 cards, all unique, 8 sevens, 2 jokers ---')
d = make_deck()
ok('106 cards', len(d) == 106, f'count={len(d)}')
ids = set(c['id'] for c in d)
ok('all ids unique', len(ids) == 106, f'unique={len(ids)}')
sevens = [c for c in d if c['rank'] == '7']
ok('8 sevens', len(sevens) == 8)
jokers = [c for c in d if c['rank'] == 'JOKER']
ok('2 jokers', len(jokers) == 2)

print('--- 2: pile-match auto-skips draw (any rank) ---')
g = make_game()
set_hands(g, ['QH','3S'], ['2H','3D','4C','5D','6D','8H','10H'])
set_floor(g, ['QC']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC'])
g.discard(0, ['QH'])
ok('phase=discard (auto-skip)', g.phase == 'discard')
ok('no draw', len(g.hands[0]) == 1)
ok('turn=1', g.turn == 1)

print('--- 3: pile-match for 7 on 7 -> war ---')
g = make_game()
set_hands(g, ['7H','5S'], ['2H','3D','4C','5D','6D','8H','10H'])
set_floor(g, ['7C']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD'])
g.discard(0, ['7H'])
ok('phase=war', g.phase == 'war')
ok('penalty=2', g.pending_penalty == 2)
ok('no draw', len(g.hands[0]) == 1)
ok('turn=1', g.turn == 1)

print('--- 4: pile-match for J on J -> play again ---')
g = make_game()
set_hands(g, ['JH','5S'], ['2H','3D','4C','5D','6D','8H','10H'])
set_floor(g, ['JC']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC'])
g.discard(0, ['JH'])
ok('phase=discard', g.phase == 'discard')
ok('turn=0 (1 jack, play again)', g.turn == 0)
ok('no draw', len(g.hands[0]) == 1)

print('--- 5: single 7 on non-7 pile: draw then war ---')
g = make_game()
set_hands(g, ['7H','5S'], ['2H','3D','4C','5D','6D','8H','10H'])
set_floor(g, ['10S']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD'])
g.discard(0, ['7H'])
ok('phase=draw', g.phase == 'draw')
ok('penalty not yet applied', g.pending_penalty == 0)
g.draw(0, 'deck')
ok('phase=war after draw', g.phase == 'war')
ok('penalty=2', g.pending_penalty == 2)
ok('A hand=2', len(g.hands[0]) == 2)
ok('turn=1', g.turn == 1)

print('--- 6: war non-7 throw -> shed + pick + turn ends ---')
g = make_game()
set_hands(g, ['7H','5S'], ['3D','4C','5D','6D','8H','10H','KS'])
set_floor(g, ['10S']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD','AS','AH'])
g.discard(0, ['7H']); g.draw(0, 'deck')
b_before = len(g.hands[1])
g.discard(1, ['3D'])
ok('war ended', g.phase == 'discard')
ok('net +1', len(g.hands[1]) == b_before - 1 + 2)
ok('penalty cleared', g.pending_penalty == 0)
ok('turn=0', g.turn == 0)

print('--- 7: 3 sevens trigger war immediately (no draw) ---')
g = make_game()
set_hands(g, ['7H','7C','7D','5S'], ['2H','3D','4C','5D','6D','8H','10H'])
set_floor(g, ['10S']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD'])
g.discard(0, ['7H','7C','7D'])
ok('phase=war', g.phase == 'war')
ok('penalty=6', g.pending_penalty == 6)
ok('hand=1', len(g.hands[0]) == 1)

print('--- 8: 7s and Js cannot be picked from open pile ---')
g = make_game()
set_hands(g, ['5H'], ['2H'])
set_floor(g, ['7H','JS','5C']); set_wild(g, 'K'); reset(g, 0)
g.phase = 'draw'; g.pending_discard = None
set_deck(g, ['AC'])
t1 = False
try: g.draw(0, 'floor', '7H')
except ValueError: t1 = True
ok('7 rejected', t1)
t2 = False
try: g.draw(0, 'floor', 'JS')
except ValueError: t2 = True
ok('J rejected', t2)
g.draw(0, 'floor', '5C')
ok('5 allowed', any(c['id'] == '5C' for c in g.hands[0]))

print('--- 9: wild rank never 7, J, or Joker (10000 deals) ---')
bad = 0
for _ in range(10000):
    g = Game(target=100, names=['A','B'])
    if g.wild_indicator['rank'] in ('7','J','JOKER'):
        bad += 1
ok('no 7/J/Joker wild', bad == 0, f'bad={bad}')

print('--- 10: declare correct - opp count capped at 40 ---')
g = make_game()
set_hands(g, ['KH','AS'], ['10H','10S','10D','10C','9H'])  # K=0 (wild), A=1 → 1 pt
set_floor(g, ['QS']); set_wild(g, 'K'); reset(g, 0)
g.declare(0)
ok('correct=True', g.round_result['correct'] == True)
ok('A=0', g.round_result['deltas'][0] == 0)
ok('B capped at 40', g.round_result['deltas'][1] == 40)

print('--- 11: declare tie -> declarer wins ---')
g = make_game()
set_hands(g, ['AS','3D'], ['AH','3C'])  # both 4 pts
set_floor(g, ['QS']); set_wild(g, 'K'); reset(g, 0)
g.declare(0)
ok('correct (tie to declarer)', g.round_result['correct'] == True)
ok('A=0', g.round_result['deltas'][0] == 0)
ok('B=4', g.round_result['deltas'][1] == 4)

print('--- 12: declare wrong (opp strictly beats) ---')
g = make_game()
set_hands(g, ['AS','3D'], ['AH'])  # A=4, B=1
set_floor(g, ['QS']); set_wild(g, 'K'); reset(g, 0)
g.declare(0)
ok('wrong', g.round_result['correct'] == False)
ok('A=+40', g.round_result['deltas'][0] == 40)
ok('B=0', g.round_result['deltas'][1] == 0)

print('--- 13: declare blocked in draw/war ---')
g = make_game()
set_hands(g, ['7H','AS'], ['AS','AH','2S','3D'])
set_floor(g, ['10S']); set_wild(g, '2'); reset(g, 0); set_deck(g, ['AC','AD'])
g.discard(0, ['7H'])
t1 = False
try: g.declare(0)
except ValueError: t1 = True
ok('declare blocked in draw', t1)
g.draw(0, 'deck')
t2 = False
try: g.declare(1)
except ValueError: t2 = True
ok('declare blocked in war', t2)

print('--- 14: open-card-7 at round start triggers war for starter ---')
found = 0
validated = 0
for _ in range(2000):
    g = Game(target=100, names=['A','B'])
    if g.floor[0]['rank'] == '7':
        found += 1
        if g.phase == 'war' and g.pending_penalty == 2 and g.turn == g.starter:
            validated += 1
ok(f'7-start found ({found} of 2000)', found > 50)
ok('every 7-start is war for starter', validated == found, f'{validated}/{found}')

print('--- 15: open-card-J at round start skips starter ---')
found = 0
validated = 0
for _ in range(3000):
    g = Game(target=100, names=['A','B'])
    if g.floor[0]['rank'] == 'J':
        found += 1
        if g.phase == 'discard' and g.turn == (g.starter ^ 1) and g.pending_penalty == 0:
            validated += 1
ok(f'J-start found ({found} of 3000)', found > 50)
ok('every J-start has turn flipped', validated == found, f'{validated}/{found}')

print('--- 16: 3+ throw empties hand -> auto-win on return ---')
g = make_game()
set_hands(g, ['5H','5D','5C'], ['2H','3D','4C','6D','8H','10H','KS'])
set_floor(g, ['QS']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD','AS','AH'])
g.discard(0, ['5H','5D','5C'])
ok('A hand=0', len(g.hands[0]) == 0)
ok('turn=1', g.turn == 1)
g.discard(1, ['2H']); g.draw(1, 'deck')
ok('A auto-wins', g.phase in ('roundover','gameover'))
ok('declarer=A', g.round_result['declarer'] == 0)

print('--- 17: empty hand + opp throws 7 -> auto-pick penalty, game continues ---')
g = make_game()
set_hands(g, ['5H','5D','5C'], ['7H','3D','4C','6D','8H','10H','KS'])
set_floor(g, ['QS']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD','AS','AH','2C','2D','2S'])
g.discard(0, ['5H','5D','5C'])
g.discard(1, ['7H']); g.draw(1, 'deck')
ok('A picked 2, game continues', len(g.hands[0]) == 2 and g.phase == 'discard')
ok('penalty cleared', g.pending_penalty == 0)
ok('turn=B', g.turn == 1)

print('--- 18: J ends 7-war -> picks penalty AND J skip applies ---')
g = make_game()
set_hands(g, ['7H','9C'], ['JC','3D','5D','6D','8H','10H','KS'])
set_floor(g, ['10S']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD','AS','AH'])
g.discard(0, ['7H']); g.draw(0, 'deck')
b_before = len(g.hands[1])
g.discard(1, ['JC'])
ok('B threw J, picked 2 (net +1)', len(g.hands[1]) == b_before - 1 + 2)
ok('turn=1 (B plays again — odd jacks)', g.turn == 1)
ok('floor=J', g.floor[0]['id'] == 'JC')

print('--- 19: 2 sevens stacked, opp counters with 1 seven, war continues ---')
g = make_game()
set_hands(g, ['7H','7C','9C'], ['7D','3D','4C','5D','6D','8H','10H'])
set_floor(g, ['10S']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD','AS','AH','2C','2D'])
g.discard(0, ['7H','7C']); g.draw(0, 'deck')
ok('penalty=4 turn=B', g.pending_penalty == 4 and g.turn == 1)
g.discard(1, ['7D'])  # matches pile (7s), auto-skip + war bounces
ok('B counter, penalty=6, turn=A', g.pending_penalty == 6 and g.turn == 0 and g.phase == 'war')

print('--- 20: serialization-free round-trip via __dict__ ---')
import copy
g = make_game()
set_hands(g, ['7H','5S'], ['3D','4C','5D','6D','8H','10H','KS'])
set_floor(g, ['10S']); set_wild(g, 'K'); reset(g, 0); set_deck(g, ['AC','AD'])
g.discard(0, ['7H'])
snap = copy.deepcopy(g.__dict__)
# Continue
g2 = make_game()
g2.__dict__.update(copy.deepcopy(snap))
ok('phase preserved', g2.phase == 'draw')
ok('pendingSevens preserved', g2.pending_sevens == 1)
g2.draw(0, 'deck')
ok('war resumes', g2.phase == 'war' and g2.pending_penalty == 2)

print('--- 21: reshuffle when deck empties ---')
g = make_game()
set_hands(g, ['5H','6S'], ['2H','3D'])
set_floor(g, ['10S']); set_wild(g, 'K'); reset(g, 0)
g.deck = []
g.graveyard = [H('AC'), H('AD'), H('AS'), H('AH'), H('2C'), H('2D')]
g.discard(0, ['5H']); g.draw(0, 'deck')
ok('A got reshuffled card', len(g.hands[0]) == 2)

print('--- 22: PROPERTY — 5000 random games complete without crashing ---')
crashed = 0
rounds_completed = 0
for trial in range(5000):
    g = Game(target=50, names=['A','B'], rng=random.Random(trial))
    steps = 0
    while g.phase != 'gameover' and steps < 2000:
        p = g.turn
        try:
            if g.phase == 'discard':
                # Random valid action: declare if eligible else throw random card(s)
                if g.can_declare(p) and random.random() < 0.05:
                    g.declare(p)
                else:
                    h = g.hands[p]
                    if not h: break  # shouldn't happen but defensive
                    # Randomly throw 1 card
                    c = random.choice(h)
                    g.discard(p, [c['id']])
            elif g.phase == 'war':
                h = g.hands[p]
                if not h: break
                # Random valid action
                sevens = [c for c in h if c['rank'] == '7']
                if sevens:
                    g.discard(p, [sevens[0]['id']])
                else:
                    c = random.choice(h)
                    g.discard(p, [c['id']])
            elif g.phase == 'draw':
                if random.random() < 0.3:
                    # Try to draw from pile (skip 7/J)
                    drawable = [c for c in g.floor if c['rank'] not in ('7','J')]
                    if drawable:
                        g.draw(p, 'floor', drawable[0]['id'])
                    else:
                        g.draw(p, 'deck')
                else:
                    g.draw(p, 'deck')
            elif g.phase == 'roundover':
                g.next_round()
                rounds_completed += 1
        except Exception as e:
            crashed += 1
            print(f'  CRASH trial {trial} step {steps}: {e}')
            break
        steps += 1
ok(f'5000 games no crashes ({rounds_completed} rounds total)', crashed == 0)

print(f'\nResults: {pass_n} pass, {fail_n} fail')
import sys
sys.exit(0 if fail_n == 0 else 1)
