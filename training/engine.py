"""
Python port of the Least Count game engine.
Mirrors the JS engine in index.html exactly. All 220 JS tests are reproduced in test_engine.py
and must pass before any training begins.
"""
import random
import copy
from typing import List, Optional, Dict, Any

RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
SUITS = ['S','H','D','C']
DECLARE_MAX = 5
SKIP_RANK = 'J'
WAR_RANK = '7'
WAR_PER_CARD = 2
BULK_NO_DRAW = 3


def rank_value(r: str) -> int:
    if r == 'JOKER': return 0
    if r == 'A': return 1
    if r in ('J','Q','K'): return 10
    return int(r)


def make_deck() -> List[Dict[str, Any]]:
    c = []
    for k in range(2):
        for s in SUITS:
            for r in RANKS:
                c.append({'id': r + s + ('a' if k == 0 else 'b'), 'rank': r, 'suit': s})
    c.append({'id': 'JOKERa', 'rank': 'JOKER', 'suit': None})
    c.append({'id': 'JOKERb', 'rank': 'JOKER', 'suit': None})
    return c


def shuffle(a: List, rng: Optional[random.Random] = None) -> List:
    rng = rng or random
    for i in range(len(a) - 1, 0, -1):
        j = rng.randint(0, i)
        a[i], a[j] = a[j], a[i]
    return a


class Game:
    def __init__(self, target: int = 100, penalty: int = 40,
                 names=None, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.target = target
        self.penalty = penalty
        self.names = names or ['Player 1', 'Player 2']
        self.scores = [0, 0]
        self.round = 0
        self.winner: Optional[int] = None
        self.phase = 'lobby'
        # Will be set in start_round
        self.hands: List[List[Dict]] = [[], []]
        self.wild_indicator: Optional[Dict] = None
        self.wild_rank: str = 'A'
        self.floor: List[Dict] = []
        self.deck: List[Dict] = []
        self.graveyard: List[Dict] = []
        self.pending_discard: Optional[List[Dict]] = None
        self.pending_jacks = 0
        self.pending_sevens = 0
        self.pending_penalty = 0
        self.war_starter: Optional[int] = None
        self.turn = 0
        self.starter = 0
        self.skip_eligible = False
        self.reveal = False
        self.round_result: Optional[Dict] = None
        self.last_action = ''
        self.start_round(0)

    def start_round(self, starter: int) -> None:
        self.round += 1
        deck = shuffle(make_deck(), self.rng)
        self.hands = [[], []]
        for _ in range(7):
            self.hands[0].append(deck.pop())
            self.hands[1].append(deck.pop())
        # Wild indicator can't be 7, J, or Joker — push them to bottom and retry
        skipped = []
        self.wild_indicator = deck.pop()
        while self.wild_indicator and self.wild_indicator['rank'] in (WAR_RANK, SKIP_RANK, 'JOKER'):
            skipped.append(self.wild_indicator)
            self.wild_indicator = deck.pop()
        for s in skipped:
            deck.insert(0, s)
        self.wild_rank = self.wild_indicator['rank']
        self.floor = [deck.pop()]
        self.deck = deck
        self.graveyard = []
        self.pending_discard = None
        self.pending_jacks = 0
        self.pending_sevens = 0
        self.war_starter = None
        self.turn = starter
        self.starter = starter
        self.skip_eligible = False
        self.reveal = False
        self.round_result = None
        if self.floor[0]['rank'] == WAR_RANK:
            self.pending_penalty = WAR_PER_CARD
            self.phase = 'war'
            self.last_action = f"Round {self.round} · open pile is a 7 — {self.names[starter]} must counter or pick {self.pending_penalty}."
        elif self.floor[0]['rank'] == SKIP_RANK:
            self.pending_penalty = 0
            self.phase = 'discard'
            self.turn = starter ^ 1
            self.last_action = f"Round {self.round} · open pile is a Jack — {self.names[starter]} skipped, {self.names[starter^1]} starts."
        else:
            self.pending_penalty = 0
            self.phase = 'discard'
            self.last_action = f"Round {self.round} · {self.names[starter]} starts."

    def card_points(self, c: Dict) -> int:
        if c['rank'] == 'JOKER': return 0
        if c['rank'] == self.wild_rank: return 0
        return rank_value(c['rank'])

    def hand_points(self, h: List[Dict]) -> int:
        return sum(self.card_points(c) for c in h)

    def floor_rank(self) -> Optional[str]:
        return self.floor[0]['rank'] if self.floor else None

    def is_valid_set(self, cards: List[Dict]) -> bool:
        if not cards: return False
        r = cards[0]['rank']
        return all(c['rank'] == r for c in cards)

    def can_declare(self, p: int) -> bool:
        return (self.phase == 'discard' and self.turn == p and
                self.pending_penalty == 0 and
                self.hand_points(self.hands[p]) <= DECLARE_MAX)

    def declare(self, p: int) -> None:
        if self.phase == 'war':
            raise ValueError(f"Answer the 7-penalty first — throw a 7 or pick up {self.pending_penalty}.")
        if not self.can_declare(p):
            raise ValueError(f"You can only declare on your turn with a hand of {DECLARE_MAX} or less.")
        self._resolve_show(p)

    def discard(self, p: int, ids: List[str]) -> Dict:
        if self.turn != p: raise ValueError("Not your move.")
        if self.phase not in ('discard', 'war'): raise ValueError("Not your move.")
        hand = self.hands[p]
        if not ids: raise ValueError("Pick at least one card.")
        # Map ids -> card refs from hand (preserving order, no duplicates)
        chosen = []
        for cid in ids:
            found = next((c for c in hand if c['id'] == cid), None)
            if found is not None and found not in chosen:
                chosen.append(found)
        if len(chosen) != len(ids): raise ValueError("Card not in hand.")
        if not self.is_valid_set(chosen): raise ValueError("Same-rank only.")
        is_seven = chosen[0]['rank'] == WAR_RANK
        is_jack = chosen[0]['rank'] == SKIP_RANK
        if self.phase == 'war' and not is_seven:
            # End war: throw + pick penalty
            jacks_thrown = len(chosen) if is_jack else 0
            self.hands[p] = [c for c in hand if c['id'] not in ids]
            self.graveyard.extend(self.floor)
            self.floor = chosen
            self.pending_discard = None
            self.pending_jacks = 0
            self.pending_sevens = 0
            self.skip_eligible = False
            pick_count = self.pending_penalty
            picked = 0
            for _ in range(pick_count):
                if not self.deck: self._reshuffle()
                if not self.deck: break
                self.hands[p].append(self.deck.pop())
                picked += 1
            self.pending_penalty = 0
            self.war_starter = None
            self.phase = 'discard'
            self.turn = p ^ ((jacks_thrown + 1) % 2)
            self.last_action = f"{self.names[p]} threw and picked {picked}."
            self._auto_win_if_empty()
            return {'warEnded': True, 'picked': picked, 'jacksThrown': jacks_thrown}
        self.hands[p] = [c for c in hand if c['id'] not in ids]
        self.pending_discard = chosen
        self.pending_jacks = len(chosen) if is_jack else 0
        self.pending_sevens = len(chosen) if is_seven else 0
        matches_pile = self.floor_rank() is not None and chosen[0]['rank'] == self.floor_rank()
        self.skip_eligible = matches_pile
        bulk = len(chosen) >= BULK_NO_DRAW
        if bulk or matches_pile:
            self.last_action = f"{self.names[p]} threw (no draw)."
            self._finish_turn()
            return {'skippedDraw': True, 'war': self.phase == 'war', 'matchedPile': matches_pile}
        self.phase = 'draw'
        self.last_action = f"{self.names[p]} threw."
        return {'skipEligible': False, 'pendingSevens': self.pending_sevens}

    def draw(self, p: int, source: str, card_id: Optional[str] = None) -> None:
        if self.phase != 'draw' or self.turn != p:
            raise ValueError("Not your draw.")
        if source == 'deck':
            if not self.deck: self._reshuffle()
            if not self.deck: raise ValueError("No cards left to draw.")
            c = self.deck.pop()
            self.hands[p].append(c)
            self.last_action = f"{self.names[p]} drew from deck."
        elif source == 'floor':
            i = next((idx for idx, c in enumerate(self.floor) if c['id'] == card_id), -1)
            if i == -1: raise ValueError("That card isn't on the pile.")
            c = self.floor[i]
            if c['rank'] == WAR_RANK: raise ValueError("Can't pick a 7 off the pile.")
            if c['rank'] == SKIP_RANK: raise ValueError("Can't pick a Jack off the pile.")
            self.floor.pop(i)
            self.hands[p].append(c)
            self.last_action = f"{self.names[p]} took from pile."
        else:
            raise ValueError("bad source")
        self._finish_turn()

    def skip_draw(self, p: int) -> None:
        if self.phase != 'draw' or self.turn != p:
            raise ValueError("Not your draw.")
        if not self.skip_eligible:
            raise ValueError("Can only skip draw if your throw matched the pile.")
        if len(self.hands[p]) < 1:
            raise ValueError("You'd have no cards.")
        self.last_action = f"{self.names[p]} skipped draw."
        self._finish_turn()

    def _finish_turn(self) -> None:
        self.graveyard.extend(self.floor)
        self.floor = self.pending_discard or []
        self.pending_discard = None
        self.skip_eligible = False
        thrower = self.turn
        sevens = self.pending_sevens
        self.pending_sevens = 0
        if sevens > 0:
            if self.war_starter is None:
                self.war_starter = thrower
            self.pending_penalty += WAR_PER_CARD * sevens
            self.pending_jacks = 0
            self.turn = thrower ^ 1
            self.phase = 'war'
            self.last_action += f" — {self.names[thrower^1]} must counter or pick {self.pending_penalty}."
            self._auto_win_if_empty()
            return
        j = self.pending_jacks
        self.pending_jacks = 0
        self.turn = thrower ^ ((j + 1) % 2)
        self.phase = 'discard'
        if j > 0:
            self.last_action += f" — {j} Jack(s)!"
        self._auto_win_if_empty()

    def _auto_win_if_empty(self) -> None:
        if self.phase in ('roundover', 'gameover'): return
        if len(self.hands[self.turn]) != 0: return
        p = self.turn
        # If a 7-war landed on empty-handed player, auto-pick penalty
        if self.phase == 'war' and self.pending_penalty > 0:
            picked = 0
            for _ in range(self.pending_penalty):
                if not self.deck: self._reshuffle()
                if not self.deck: break
                self.hands[p].append(self.deck.pop())
                picked += 1
            self.pending_penalty = 0
            self.war_starter = None
            self.pending_sevens = 0
            self.phase = 'discard'
            self.turn = p ^ 1
            self.last_action = f"{self.names[p]} auto-picked {picked} from 7-penalty."
            return
        # Auto-win
        self.pending_penalty = 0
        self.pending_sevens = 0
        self.pending_jacks = 0
        self.war_starter = None
        self.pending_discard = None
        self._resolve_show(p)
        self.last_action = f"{self.names[p]} ran out of cards — auto-win!"

    def _reshuffle(self) -> None:
        if not self.graveyard: return
        self.deck = shuffle(self.graveyard, self.rng)
        self.graveyard = []

    def _resolve_show(self, declarer: int) -> None:
        t = [self.hand_points(self.hands[0]), self.hand_points(self.hands[1])]
        o = declarer ^ 1
        correct = t[declarer] <= t[o]
        d = [0, 0]
        cap = self.penalty
        if correct:
            d[declarer] = 0
            d[o] = min(t[o], cap)
        else:
            d[declarer] = cap
            d[o] = 0
        self.scores[0] += d[0]
        self.scores[1] += d[1]
        self.round_result = {
            'declarer': declarer, 'totals': t, 'correct': correct, 'deltas': d
        }
        self.reveal = True
        self.phase = 'roundover'
        self.last_action = f"{self.names[declarer]} declared with {t[declarer]} — {'correct' if correct else 'WRONG'}."
        if self.scores[0] >= self.target or self.scores[1] >= self.target:
            self.winner = 0 if self.scores[0] < self.scores[1] else 1
            self.phase = 'gameover'

    def next_round(self) -> None:
        if self.phase == 'gameover': return
        self.start_round(self.starter ^ 1)

    def view_for(self, p: int) -> Dict[str, Any]:
        opp = p ^ 1
        return {
            'you': p,
            'names': list(self.names),
            'target': self.target,
            'penalty': self.penalty,
            'round': self.round,
            'scores': list(self.scores),
            'yourHand': [{'id': c['id'], 'rank': c['rank'], 'suit': c['suit'],
                          'pts': self.card_points(c)} for c in self.hands[p]],
            'yourPoints': self.hand_points(self.hands[p]),
            'oppHandCount': len(self.hands[opp]),
            'oppHand': ([{'id': c['id'], 'rank': c['rank'], 'suit': c['suit'],
                          'pts': self.card_points(c)} for c in self.hands[opp]]
                        if self.reveal else None),
            'oppPoints': self.hand_points(self.hands[opp]) if self.reveal else None,
            'floor': [{'id': c['id'], 'rank': c['rank'], 'suit': c['suit'],
                       'pts': self.card_points(c)} for c in self.floor],
            'pendingDiscard': ([{'id': c['id'], 'rank': c['rank'], 'suit': c['suit'],
                                 'pts': self.card_points(c)} for c in self.pending_discard]
                               if self.pending_discard else None),
            'deckCount': len(self.deck),
            'wildRank': self.wild_rank,
            'wildIndicator': self.wild_indicator,
            'turn': self.turn,
            'yourTurn': self.turn == p,
            'phase': self.phase,
            'skipEligible': self.skip_eligible,
            'pendingPenalty': self.pending_penalty,
            'pendingSevens': self.pending_sevens,
            'canDeclare': self.can_declare(p),
            'reveal': self.reveal,
            'roundResult': self.round_result,
            'winner': self.winner,
            'lastAction': self.last_action,
            'declareMax': DECLARE_MAX,
        }
