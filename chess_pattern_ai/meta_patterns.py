#!/usr/bin/env python3
"""
Universal gameplay elements: the third level of pattern data

The pattern stores (pattern_store.py) hold, per game, what was learned
(specific) and the facts several games share (generic). Generic facts are
still tied to how a game is played: "action 1 moves up", "the opponent's
clicks drop along a column". Here the facts of games, and the
transformations that solve ARC tasks, are described by the element of
play they are an instance of, independent of how it is carried out:

    gravity       things move one way by themselves until blocked
                  (a falling piece, a disc dropped in a column, ARC
                  gravity and objects slid until blocked)
    alignment     a line of k cells (a line of marks that wins or loses, a
                  full row that scores, ARC rays, lines extended or
                  connected, objects lined up)
    rotation      a shape turned by a quarter or half turn (a piece that
                  turns, ARC rotations)
    reflection    a shape mirrored, or completed to a symmetry (an action
                  that mirrors, ARC reflections and symmetry completion)
    shape_match   one shape made equal to, or copied from, another (a
                  player that must take a target's shape, ARC templates
                  stamped or placed around objects)
    removal       cells of a value removed (a level won when no cell holds
                  a value, ARC objects removed)
    reach         one object brought to touch another (a level won by
                  touching an object, ARC objects approaching another)
    movement      an object moved step by step in four directions (actions
                  that move the player, ARC objects shifted or moved)
    opponents     other players act by the same rules as the player

Each instance is recorded with its mechanism (how it was carried out: for
games the kind of fact, such as 'drift', 'drop' or 'line_goal:any'; for
ARC 'arc:' and the solver's family) and its parameters. An element is a
meta-pattern when its instances come from at least MIN_MECHANISMS
different mechanisms: seen in only one way, it is a fact about one kind of
game or task, not a universal element.

Using elements:
- in games (ObjectAgent.use_meta), where neither the game's own evidence
  nor a generic fact covers the point: gravity in a direction makes a
  drift accepted from one observation and click effects tried as drops in
  that direction first; alignment makes lines of the agent's marks (or,
  in a game whose pieces come to rest, lines of any cells across the
  direction of gravity) candidate goals, and lines of the other players'
  marks candidate losses; rotation and reflection make a turn or mirror of
  the agent's pattern accepted from one observation; shape_match makes
  "the agent takes another object's shape" a candidate goal once the
  agent has grown; reach makes the only object of its pattern a candidate
  goal (see ObjectAgent._apply_meta)
- in ARC (evaluate_meta_arc.py): the pattern solver's families are
  searched in the order of the weight of the elements they embody
  (FAMILY_ELEMENTS), and ties among equally simple patterns are broken by
  the weight of the elements a pattern's name shows (arc_elements)

MetaStore(<dir>) keeps <dir>/meta.json: the instances by element (from
games, through add_game, and from ARC tasks, through add_arc) and, for
each element used in a new game, whether that game's own learning ended
with an instance of it.
"""

import json
import os
from collections import Counter
from typing import Dict, List

ELEMENTS = ('gravity', 'alignment', 'rotation', 'reflection', 'shape_match', 'removal', 'reach', 'movement',
            'opponents')
MIN_MECHANISMS = 2
TURNS = ('turn right', 'turn left', 'half turn')
MIRRORS = ('mirror left-right', 'mirror up-down', 'transpose')

# the elements each family of arc_pattern_solver can embody (its description in that module)
FAMILY_ELEMENTS = {
    'B': ('rotation', 'reflection'),                    # a cropped region, optionally turned or mirrored
    'C': ('rotation', 'reflection', 'gravity', 'alignment', 'removal'),
    'D': ('movement', 'gravity'),                       # shifted, or slid until blocked
    'F': ('reach', 'shape_match'),                      # approach until blocked; a template stamped
    'K': ('alignment',),                                # rays
    'M': ('movement',),
    'N': ('reflection', 'rotation'),                    # symmetry completed
    'O': ('shape_match', 'alignment'),                  # cells placed around objects; objects lined up
    'Q': ('reflection',),
}

# words in an ARC pattern's name that show an element
ARC_WORDS = {
    'gravity': ('gravity:', 'slid'),
    'alignment': ('rays from', 'ray:', 'extend_lines', 'connect:', 'line up'),
    'rotation': ('rot90', 'rot180', 'rot270', 'anti_transpose'),
    'reflection': ('flip_', 'transpose', 'mirror', 'symmetry', 'reflection'),
    'shape_match': ('stamp', 'template', 'cells placed around'),
    'removal': ('remove_', 'keep_'),
    'reach': ('approach',),
    'movement': ('shift[', ' move '),
}


def arc_elements(family: str, name: str) -> set:
    """The elements an ARC pattern's name shows"""
    low = name.lower()
    return {e for e, words in ARC_WORDS.items() if any(w in low for w in words)}


def game_elements(facts: Dict[tuple, dict]) -> List[tuple]:
    """(element, mechanism, parameters) for the facts a game's agent learned (ObjectAgent.facts)"""
    out = []
    for key, params in facts.items():
        kind = key[0]
        if kind == 'drift':
            out.append(('gravity', 'drift', {'d': list(key[2])}))
        elif kind in ('own_click', 'other_click') and key[2] == 'drop':
            out.append(('gravity', 'drop', {'d': list(key[3])}))
        elif kind in ('line_goal', 'line_loss'):
            out.append(('alignment', f'{kind}:{key[1]}',
                        {'role': key[1], 'd': list(key[2]) if key[2] else None, 'k': params.get('k', []),
                         'full': bool(params.get('full')), 'loss': kind == 'line_loss'}))
        elif kind == 'transform':
            if key[2] in TURNS:
                out.append(('rotation', 'turn', {'op': key[2]}))
            elif key[2] in MIRRORS:
                out.append(('reflection', 'turn', {'op': key[2]}))
        elif kind == 'shape_goal':
            out.append(('shape_match', 'shape_goal', {}))
        elif kind == 'absent_goal':
            out.append(('removal', 'absent_goal', {}))
        elif kind == 'contact_goal':
            out.append(('reach', 'contact_goal', {'kind': key[1]}))
        elif kind == 'direction':
            out.append(('movement', 'direction', {}))
        elif kind == 'others_like_me':
            out.append(('opponents', 'others_like_me', {}))
    # one instance per (element, mechanism, parameters)
    seen, unique = set(), []
    for e in out:
        k = json.dumps(e, sort_keys=True)
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return unique


class MetaStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._file = os.path.join(path, 'meta.json')
        self.data = {'instances': {}, 'use': {}}
        if os.path.exists(self._file):
            with open(self._file) as f:
                self.data = json.load(f)

    def _save(self):
        with open(self._file, 'w') as f:
            json.dump(self.data, f, indent=1)

    def add_game(self, game: str, facts: Dict[tuple, dict]):
        """Replace the game's instances with those of its facts"""
        self.data['instances'][f'game:{game}'] = [list(e) for e in game_elements(facts)]
        self._save()

    def add_arc(self, task: str, family: str, name: str):
        """The elements of the pattern that solved an ARC task"""
        self.data['instances'][f'arc:{task}'] = [[e, f'arc:{family}', {}] for e in sorted(arc_elements(family, name))]
        self._save()

    def record_use(self, game: str, used: set, facts: Dict[tuple, dict]):
        found = {e for e, _, _ in game_elements(facts)}
        for e in used:
            self.data['use'].setdefault(e, {})[game] = 'confirmed' if e in found else 'not confirmed'
        self._save()

    def elements(self, exclude=()) -> Dict[str, dict]:
        """
        The meta-patterns: elements with instances from at least MIN_MECHANISMS mechanisms (sources in
        `exclude`, such as the game being tested, left out): their mechanisms, sources and parameters
        """
        by = {}
        for source, instances in self.data['instances'].items():
            if source in exclude:
                continue
            for element, mechanism, params in instances:
                e = by.setdefault(element, {'mechanisms': Counter(), 'sources': set(), 'params': []})
                e['mechanisms'][mechanism] += 1
                e['sources'].add(source)
                e['params'].append(dict(params, mechanism=mechanism))
        out = {}
        for element, e in by.items():
            uses = Counter(self.data['use'].get(element, {}).values())
            if uses['not confirmed'] > uses['confirmed']:
                continue                        # used in new games more often without confirmation than with
            if len(e['mechanisms']) >= MIN_MECHANISMS:
                out[element] = {'mechanisms': dict(e['mechanisms']), 'sources': sorted(e['sources']),
                                'games': sum(s.startswith('game:') for s in e['sources']),
                                'arc_tasks': sum(s.startswith('arc:') for s in e['sources']),
                                'params': e['params']}
        return out

    def weights(self, exclude=()) -> Dict[str, float]:
        """Each meta-pattern's weight: the share of the sources (games, ARC tasks) that show it"""
        sources = [s for s in self.data['instances'] if s not in exclude]
        return {e: len(v['sources']) / max(1, len(sources)) for e, v in self.elements(exclude).items()}

    def family_order(self, families: List[str], exclude=()) -> List[str]:
        """The ARC families, those embodying the heaviest meta-pattern first (ties: listing order)"""
        w = self.weights(exclude)
        return sorted(families, key=lambda f: (-max([w.get(e, 0.0) for e in FAMILY_ELEMENTS.get(f, ())] or [0.0]),
                                               families.index(f)))
