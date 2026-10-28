#!/usr/bin/env python3
"""
Two stores of learned patterns for the object agent (object_agent.ObjectAgent)

- specific (<dir>/specific/<game>.json): one entry per game, written when a
  session of that game ends:
    facts   what was learned, with each value replaced by its role in the
            game (ObjectAgent.facts: 'mine', 'theirs', 'empty', 'wall', ...),
            so that games drawn in different colours can be compared
    state   the agent's learned rules, models, goals and weights with the
            game's own values (ObjectAgent.export_state), to go on learning
            the same game in a later session
- generic (<dir>/generic.json): meta-patterns, the facts found in at least
  MIN_GAMES games' specific entries. Each holds
    games     the games it was found in
    params    the games' parameters merged (line lengths: all those seen,
              and whether each was the grid's extent along the line in all
              the games; score weights: the mean of the games' means and
              spreads)
    context   the condition facts (CONDITIONS: how the agent acts, other
              players, drift, the agent's own clicks) common to those games; a pattern is used
              in a new game only once all of them hold there
    use       for each new game it was used in, whether that game's own
              learning ended with the same fact ('confirmed'), with
              another fact of its kind only ('not confirmed': for a line
              goal, wins formed other lines), or with neither ('no evidence')
  A pattern not confirmed more often than confirmed is no longer offered
  (generic()).

PatternStore.start(game, agent): a game with a specific entry goes on from
its state; generic patterns are offered in any case (the agent uses them
only where the game's own evidence is missing). PatternStore.save(game,
agent, episodes) writes the game's entry, records how the generic patterns
the agent used turned out, and rebuilds the generic store from all specific
entries.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from object_agent import ClickModel, ObjectAgent

MIN_GAMES = 2
CONDITIONS = ('acts_by', 'players', 'drift', 'own_click')
SEEN_KEPT = 16              # clicks a stored click model keeps (of those it has seen)


# ------------------------------------------------------------ encoding
def encode(x):
    """JSON for the agent's state: tuples, sets, Counters, dicts with any keys, arrays, click models"""
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, np.ndarray):
        return {'array': x.tolist(), 'dtype': str(x.dtype)}
    if isinstance(x, tuple):
        return {'tuple': [encode(v) for v in x]}
    if isinstance(x, frozenset):
        return {'frozenset': [encode(v) for v in x]}
    if isinstance(x, set):
        return {'set': [encode(v) for v in x]}
    if isinstance(x, Counter):
        return {'counter': [[encode(k), encode(v)] for k, v in x.items()]}
    if isinstance(x, dict):
        return {'dict': [[encode(k), encode(v)] for k, v in x.items()]}
    if isinstance(x, list):
        return [encode(v) for v in x]
    if isinstance(x, ClickModel):
        fields = dict(vars(x))
        fields['seen'] = fields['seen'][-SEEN_KEPT:]
        return {'click_model': encode(fields)}
    raise TypeError(f'cannot store {type(x)}')


def decode(x):
    if isinstance(x, list):
        return [decode(v) for v in x]
    if not isinstance(x, dict):
        return x
    if 'array' in x:
        return np.array(x['array'], dtype=x['dtype'])
    (tag, v), = x.items()
    if tag == 'tuple':
        return tuple(decode(e) for e in v)
    if tag == 'frozenset':
        return frozenset(decode(e) for e in v)
    if tag == 'set':
        return {decode(e) for e in v}
    if tag == 'counter':
        return Counter({decode(k): decode(n) for k, n in v})
    if tag == 'dict':
        return {decode(k): decode(e) for k, e in v}
    if tag == 'click_model':
        m = ClickModel.__new__(ClickModel)
        m.__dict__.update(ClickModel().__dict__)
        m.__dict__.update(decode(v))
        return m
    raise ValueError(f'unknown tag {tag}')


def _key_json(key) -> str:
    return json.dumps(key)


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def _merge(params: List[dict]) -> dict:
    out = {}
    for p in params:
        for name, value in p.items():
            out.setdefault(name, []).append(value)
    merged = {}
    for name, values in out.items():
        if name == 'k':
            merged[name] = sorted({k for ks in values for k in ks})
        elif name == 'full':
            merged[name] = all(values)
        else:
            merged[name] = np.mean(np.array(values, dtype=float), axis=0).tolist()
    return merged


def _tested(key, facts) -> bool:
    """Whether the game learned something of the key's kind (so that the key's absence counts against it)"""
    if key[0] == 'direction':
        return any(k[0] in ('direction', 'no_direction') and k[1] == key[1] for k in facts)
    needs = {'loss_like_goal': ('line_goal', 'line_loss'), 'others_like_me': ('other_click',)}.get(key[0], (key[0],))
    return all(any(k[0] == n and (n not in ('drift', 'transform') or k[1] == key[1]) for k in facts) for n in needs)


class PatternStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.join(path, 'specific'), exist_ok=True)
        self._generic_file = os.path.join(path, 'generic.json')

    # ------------------------------------------------------- specific
    def _file(self, game) -> str:
        return os.path.join(self.path, 'specific', f'{game}.json')

    def games(self) -> List[str]:
        return sorted(f[:-5] for f in os.listdir(os.path.join(self.path, 'specific')) if f.endswith('.json'))

    def load(self, game) -> Optional[dict]:
        """The game's specific entry (facts with tuple keys, state decoded), if there is one"""
        if not os.path.exists(self._file(game)):
            return None
        with open(self._file(game)) as f:
            entry = json.load(f)
        entry['facts'] = {_tuples(f['key']): f['params'] for f in entry['facts']}
        entry['state'] = decode(entry['state'])
        return entry

    def start(self, game, agent) -> str:
        """Prepare the agent for a session of the game: 'specific' (its own entry) or 'generic'"""
        entry = self.load(game)
        if entry is not None:
            agent.import_state(entry['state'])
        agent.use_generic(self.generic())
        return 'specific' if entry is not None else 'generic'

    @staticmethod
    def _facts_json(facts) -> List[dict]:
        return [{'key': list(k), 'params': p} for k, p in sorted(facts.items(), key=str)]

    def save(self, game, agent, episodes: int = 0):
        """Write the game's entry, record how the generic patterns used turned out, rebuild the generic store"""
        old = self.load(game)
        facts = agent.facts()
        entry = {'game': game,
                 'sessions': (old['sessions'] if old else 0) + 1,
                 'episodes': (old['episodes'] if old else 0) + episodes,
                 'facts': self._facts_json(facts),
                 'state': encode(agent.export_state())}
        with open(self._file(game), 'w') as f:
            json.dump(entry, f)
        generic = self._read_generic()
        for key in agent.priors_applied:
            use = generic['use'].setdefault(_key_json(key), {})
            use[game] = 'confirmed' if key in facts else 'not confirmed' if _tested(key, facts) else 'no evidence'
        self._write_generic(generic['use'])

    # -------------------------------------------------------- generic
    def _read_generic(self) -> dict:
        if not os.path.exists(self._generic_file):
            return {'patterns': [], 'use': {}}
        with open(self._generic_file) as f:
            return json.load(f)

    def _write_generic(self, use: dict):
        found: Dict[str, dict] = {}
        for game in self.games():
            for key, params in self.load(game)['facts'].items():
                e = found.setdefault(_key_json(key), {'key': list(key), 'games': [], 'params': []})
                e['games'].append(game)
                e['params'].append(params)
        conditions = {game: {k for k in self.load(game)['facts'] if k[0] in CONDITIONS} for game in self.games()}
        patterns = []
        for kj, e in sorted(found.items()):
            if len(e['games']) < MIN_GAMES:
                continue
            key = _tuples(e['key'])
            context = set.intersection(*(conditions[g] for g in e['games'])) - {key}
            patterns.append({'key': e['key'], 'games': e['games'], 'params': _merge(e['params']),
                             'context': sorted((list(c) for c in context), key=str), 'use': use.get(kj, {})})
        with open(self._generic_file, 'w') as f:
            json.dump({'patterns': patterns, 'use': use}, f, indent=1)

    def refresh(self):
        """Recompute every game's facts from its stored state (after the facts' definition changed)"""
        for game in self.games():
            agent = ObjectAgent()
            agent.import_state(self.load(game)['state'])
            with open(self._file(game)) as f:
                entry = json.load(f)
            entry['facts'] = self._facts_json(agent.facts())
            with open(self._file(game), 'w') as f:
                json.dump(entry, f)
        self.rebuild()

    def rebuild(self):
        self._write_generic(self._read_generic()['use'])

    def generic(self) -> List[dict]:
        """The generic patterns offered to a new game (keys and context as tuples)"""
        out = []
        for p in self._read_generic()['patterns']:
            uses = Counter(p['use'].values())
            if uses['not confirmed'] > uses['confirmed']:
                continue
            out.append({'key': _tuples(p['key']), 'games': p['games'], 'params': p['params'],
                        'context': [_tuples(c) for c in p['context']]})
        return out
