#!/usr/bin/env python3
"""Unit tests for arc3_agent and evaluate_arc3 (run: python3 -m unittest test_arc3_agent)"""

import random
import unittest

import numpy as np

from arc3_agent import Arc3Agent, GridReader, cell_grid, find_scale
from object_agent import ObjectAgent

try:
    from arcengine import ARCBaseGame, ActionInput, BlockingMode, Camera, GameAction, Level, Sprite
except ImportError:         # the ARC-AGI toolkit is not installed
    ARCBaseGame = None

WALLS = [
    [5, 5, 5, 5, 5, 5, 5, 5],
    [5, -1, -1, -1, 5, -1, -1, 5],
    [5, -1, 5, -1, 5, -1, 5, 5],
    [5, -1, 5, -1, -1, -1, -1, 5],
    [5, -1, 5, 5, 5, 5, -1, 5],
    [5, -1, -1, -2, -1, 5, -1, 5],      # -2: a wall that is not drawn
    [5, 5, 5, 5, -1, -1, -1, 5],
    [5, 5, 5, 5, 5, 5, 5, 5],
]


def maze_game():
    """A one-level maze: the player (8) must step onto the exit (9); one wall is not drawn"""
    class Maze(ARCBaseGame):
        def __init__(self):
            sprites = [Sprite(pixels=WALLS, name='walls', blocking=BlockingMode.PIXEL_PERFECT, layer=-1),
                       Sprite(pixels=[[8]], name='player', blocking=BlockingMode.BOUNDING_BOX).set_position(1, 1),
                       Sprite(pixels=[[9]], name='exit', blocking=BlockingMode.BOUNDING_BOX).set_position(6, 6)]
            super().__init__(game_id='maze', levels=[Level(sprites=sprites, grid_size=(8, 8))],
                             camera=Camera(background=0, letter_box=3))

        def step(self):
            dx, dy = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                      GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}.get(self.action.id, (0, 0))
            hit = self.try_move('player', dx, dy)
            if hit and any(s.name == 'exit' for s in hit):
                self.win()
            self.complete_action()
    return Maze()


def merge_game():
    """
    The player (9) absorbs pieces it walks into; the level is won when its
    shape equals the target's, lost when no piece is left otherwise
    """
    import numpy as np

    class MergeGame(ARCBaseGame):
        def __init__(self):
            border = [[5] * 10] + [[5] + [-1] * 8 + [5] for _ in range(8)] + [[5] * 10]
            sprites = [Sprite(pixels=border, name='walls', blocking=BlockingMode.PIXEL_PERFECT, layer=-1),
                       Sprite(pixels=[[9]], name='player', blocking=BlockingMode.PIXEL_PERFECT,
                              tags=['merge']).set_position(2, 7),
                       Sprite(pixels=[[8], [8]], name='piece', blocking=BlockingMode.PIXEL_PERFECT,
                              tags=['merge']).set_position(5, 4),
                       Sprite(pixels=[[-1, 8], [9, 8]], name='target', blocking=BlockingMode.PIXEL_PERFECT,
                              tags=['target']).set_position(7, 1)]
            super().__init__(game_id='merge', levels=[Level(sprites=sprites, grid_size=(10, 10))],
                             camera=Camera(width=10, height=10, background=1, letter_box=3))

        def on_set_level(self, level):
            self._player = level.get_sprites_by_name('player')[0]
            self._target = level.get_sprites_by_tag('target')[0]

        def step(self):
            dx, dy = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                      GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}.get(self.action.id, (0, 0))
            if dx or dy:
                for other in self.try_move('player', dx, dy):
                    if 'merge' in other.tags:
                        old, self._player = self._player, self._player.merge(other)
                        self.current_level.remove_sprite(other)
                        self.current_level.remove_sprite(old)
                        self.current_level.add_sprite(self._player)
                        self._player.move(dx, dy)
            if np.array_equal(self.get_pixels_at_sprite(self._player), self.get_pixels_at_sprite(self._target)):
                self.win()
            elif len(self.current_level.get_sprites_by_tag('merge')) <= 1:
                self.lose()
            self.complete_action()
    return MergeGame()


class LocalEnv:
    """The reset/step interface of the toolkit's environment wrappers, over a game object"""

    def __init__(self, game):
        self.game = game

    def reset(self):
        return self.game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

    def step(self, action, data=None):
        return self.game.perform_action(ActionInput(id=action, data=data or {}), raw=True)


class TestGrid(unittest.TestCase):
    def test_drop_target(self):
        from object_agent import ClickModel
        g = np.array([[5, 5, 5], [5, 0, 5], [5, 0, 5], [5, 2, 5], [5, 5, 5]])
        self.assertEqual(ClickModel.drop_target(g, (1, 1), (1, 0), 0), (2, 1))     # onto the 2
        self.assertEqual(ClickModel.drop_target(g, (3, 1), (1, 0), 0), (2, 1))     # a click on the 2 too
        g[2, 1] = g[1, 1] = 2
        self.assertIsNone(ClickModel.drop_target(g, (1, 1), (1, 0), 0))            # a full column

    def test_scale_letterbox_and_status_bar(self):
        rng = np.random.default_rng(0)
        cells = rng.integers(0, 10, size=(12, 12))
        frame = np.full((64, 64), 3)
        frame[2:62, 2:62] = np.kron(cells, np.ones((5, 5), dtype=int))
        frame[0:2, :] = 6                       # a status bar over the top edge
        frame[2:4, 10:20] = 6                   # ... reaching into the outer ring of cells
        self.assertEqual(find_scale(frame), (5, 2, 2))
        self.assertTrue(np.array_equal(cell_grid(frame), cells))

    def test_reader_finds_new_scale(self):
        read = GridReader()
        a = np.kron(np.arange(64).reshape(8, 8) % 7, np.ones((8, 8), dtype=int))
        self.assertEqual(read(a).shape, (8, 8))
        b = np.kron(np.arange(256).reshape(16, 16) % 5, np.ones((4, 4), dtype=int))
        self.assertEqual(read(b).shape, (16, 16))


@unittest.skipIf(ARCBaseGame is None, 'arcengine is not installed')
class TestAgent(unittest.TestCase):
    def test_learns_maze_with_hidden_wall(self):
        from evaluate_arc3 import play
        agent = Arc3Agent(random.Random(0))
        results = []
        for _ in range(3):
            agent.new_episode()
            results.append(play(LocalEnv(maze_game()), agent.act, 300, agent))
        self.assertEqual([r[0] for r in results], [1, 1, 1])
        self.assertEqual(agent.agent_values, {8})
        self.assertEqual(agent.directions, {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)})
        self.assertEqual(agent.win_keys, {(8, 9)})
        self.assertIsNotNone(agent.model())
        # once learned, the shortest route (10 moves) is taken
        self.assertEqual(results[2][2], [10])


@unittest.skipIf(ARCBaseGame is None, 'arcengine is not installed')
class TestObjectAgent(unittest.TestCase):
    def episodes(self, make, n, budget):
        from evaluate_arc3 import play
        agent = ObjectAgent(random.Random(0))
        results = []
        for _ in range(n):
            agent.new_episode()
            results.append(play(LocalEnv(make()), agent.act, budget, agent))
        return agent, results

    def test_maze_contact_goal(self):
        agent, results = self.episodes(maze_game, 3, 300)
        self.assertEqual([r[0] for r in results], [1, 1, 1])
        self.assertEqual(agent.bg, 0)
        self.assertEqual(agent.directions, {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)})
        self.assertEqual(agent.win_contacts, {((9,),)})
        self.assertEqual(results[2][2], [10])

    def test_merge_shape_goal(self):
        agent, results = self.episodes(merge_game, 3, 1000)
        self.assertEqual([r[0] for r in results], [1, 1, 1])
        self.assertTrue(agent.shape_goal)
        self.assertEqual(agent.rules[((8,), (8,))].most_common(1)[0][0], 'merge')
        self.assertEqual(agent._walls(), {5})
        # once learned: the player walks to the piece's left side and absorbs it (no losses)
        self.assertEqual(results[2][3], 0)


    def test_exit_and_shape_levels(self):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play
        agent = ObjectAgent(random.Random(0))
        results = []
        for _ in range(2):
            agent.new_episode()
            results.append(play(local_env(GAMES['exit-shape']()), agent.act, 1000, agent))
        self.assertEqual([r[0] for r in results], [4, 4])
        self.assertTrue(agent.win_contacts)              # levels won by reaching the exit
        self.assertTrue(agent.shape_goal)                # ... and by taking the target's shape
        self.assertEqual(results[1][3], 0)

    def test_shape_and_touch(self):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play
        agent = ObjectAgent(random.Random(0))
        results = []
        for _ in range(2):
            agent.new_episode()
            results.append(play(local_env(GAMES['shape-touch']()), agent.act, 1000, agent))
        self.assertEqual([r[0] for r in results], [3, 3])
        self.assertTrue(agent.shape_contact)             # touching the target while having its shape
        self.assertIn(('shape_contact_goal',), agent.facts())
        self.assertEqual(sum(r[3] for r in results), 0)

@unittest.skipIf(ARCBaseGame is None, 'arcengine is not installed')
class TestClicks(unittest.TestCase):
    def episodes(self, name, n, budget):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play
        agent = ObjectAgent(random.Random(0))
        results = []
        for _ in range(n):
            agent.new_episode()
            results.append(play(local_env(GAMES[name]()), agent.act, budget, agent))
        return agent, results

    def test_clear_red(self):
        agent, results = self.episodes('clear', 2, 300)
        self.assertEqual([r[0] for r in results], [3, 3])
        self.assertEqual(agent.click_models[2].kind(), 'map')
        self.assertEqual(agent.click_models[2].table, {2: 0})
        self.assertEqual(agent.absent_goals, {2})
        self.assertEqual(results[1][2], [2, 3, 4])          # one click per red object

    def test_lights_out(self):
        agent, results = self.episodes('lights', 2, 300)
        self.assertEqual([r[0] for r in results], [3, 3])
        self.assertEqual(agent.click_models[11].toggles(), (1, 11))
        self.assertEqual(agent.click_models[11].offsets, {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)})
        self.assertEqual(agent.absent_goals, {11})
        self.assertEqual(results[1][3], 0)

    def test_tic_tac_toe(self):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play
        agent = ObjectAgent(random.Random(0))
        results = []
        for ep in range(6):
            agent.new_episode()
            results.append(play(local_env(GAMES['tictactoe'](seed=ep)), agent.act, 300, agent))
        self.assertEqual(len(agent.others), 1)                  # one other player, seen in the second frame
        self.assertEqual(agent.line_goals, {(2, 3, None)})        # three of the agent's marks in a line
        self.assertIn((1, 3, None), agent.line_losses)            # three of the opponent's
        self.assertEqual([r[0] for r in results[1:]], [3] * 5)
        self.assertEqual(sum(r[3] for r in results[1:]), 0)       # no game lost after the first episode

    def test_tetris_frames_and_score(self):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play
        from object_agent import ANY
        agent = ObjectAgent(random.Random(0))
        scores = []
        for ep in range(10):
            agent.new_episode()
            scores.append(play(local_env(GAMES['tetris'](seed=ep)), agent.act, 3000, agent, stop_on_loss=True)[4])
        self.assertEqual(agent._drift_dir(1), (1, 0))                       # pieces fall by themselves
        self.assertEqual(agent.transform_ops[1].most_common(1)[0][0], ('turn right', (0, 0)))
        self.assertIn((ANY, 16, (0, 1)), agent.line_goals)                 # a full row scores
        self.assertGreater(sum(scores), 0)

    def test_connect_four(self):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play
        agent = ObjectAgent(random.Random(0))
        results = []
        for ep in range(6):
            agent.new_episode()
            env = local_env(GAMES['connect4'](seed=ep))
            play(env, agent.act, 300, agent, stop_on_loss=True)
            results.append(env.game.result)
        self.assertEqual(agent.click_models[0].kind(), 'drop')            # a click drops into its column
        self.assertEqual(agent.click_models[0].drop_dir(), (1, 0))
        self.assertIn((2, 4, None), agent.line_goals)
        self.assertGreaterEqual(results[2:].count('win'), 3)

    def test_empty_board_scale(self):
        from arc3_games import GAMES, local_env
        env = local_env(GAMES['connect4'](seed=0))
        self.assertEqual(find_scale(np.asarray(env.reset().frame[-1]))[0], 7)    # not a larger, empty-looking one

    def test_grid_reader_pixel(self):
        read = GridReader()
        frame = np.full((64, 64), 3)
        frame[2:62, 2:62] = np.kron(np.arange(25).reshape(5, 5) % 4, np.ones((12, 12), dtype=int))
        read(frame)
        self.assertEqual(read.pixel((0, 0)), (8, 8))
        self.assertEqual(read.pixel((4, 2)), (32, 56))



@unittest.skipIf(ARCBaseGame is None, 'arcengine is not installed')
class TestPatternStore(unittest.TestCase):
    @staticmethod
    def session(agent, colours, n, rows=3, k=3):
        from arc3_games import KInARow, local_env
        from evaluate_arc3 import play
        results = []
        for i in range(n):
            env = local_env(KInARow(seed=i, rows=rows, cols=rows, k=k, drop=False, colours=colours))
            agent.new_episode()
            play(env, agent.act, 100, agent, stop_on_loss=True)
            results.append(env.game.result)
        return results

    def test_state_round_trip(self):
        import json
        from pattern_store import decode, encode
        agent = ObjectAgent(random.Random(0))
        self.session(agent, (0, 9, 6, 4), 6)
        again = ObjectAgent(random.Random(0))
        again.import_state(decode(json.loads(json.dumps(encode(agent.export_state())))))
        self.assertEqual(again.facts(), agent.facts())
        self.assertEqual(again.line_goals, agent.line_goals)
        self.assertEqual(again.click_models[0].kind(), agent.click_models[0].kind())

    def test_generic_from_two_games(self):
        import tempfile
        from pattern_store import PatternStore
        with tempfile.TemporaryDirectory() as d:
            store = PatternStore(d)
            for name, colours, rows in (('a', (0, 9, 6, 4), 3), ('b', (0, 2, 1, 5), 4)):
                agent = ObjectAgent(random.Random(0))
                self.assertEqual(store.start(name, agent), 'generic')
                self.session(agent, colours, 8, rows)
                store.save(name, agent, 8)
            self.assertEqual(store.games(), ['a', 'b'])
            generic = {p['key']: p for p in store.generic()}
            # the same patterns in both games, though drawn in other colours
            self.assertIn(('line_goal', 'mine', None), generic)
            self.assertIn(('players', 1), generic[('line_goal', 'mine', None)]['context'])
            self.assertIn(('others_like_me',), generic)
            # a third game, in colours of its own: the goal is taken to be a line of its marks (7) at once
            agent = ObjectAgent(random.Random(0))
            store.start('c', agent)
            self.session(agent, (0, 7, 8, 5), 1)
            self.assertIn(('line_goal', 'mine', None), agent.priors_applied)
            self.assertIn((7, 3, None), agent.line_goals)
            # a game played again goes on from its own entry
            agent = ObjectAgent(random.Random(0))
            self.assertEqual(store.start('a', agent), 'specific')
            self.assertIn((9, 3, None), agent.line_goals)


    def test_directions_and_contact_goal(self):
        import tempfile
        from arc3_games import Maze, local_env
        from evaluate_arc3 import play
        from pattern_store import PatternStore
        with tempfile.TemporaryDirectory() as d:
            store = PatternStore(d)
            for name, colours in (('a', (0, 5, 8, 9, 7)), ('b', (1, 4, 12, 14, 6))):
                agent = ObjectAgent(random.Random(0))
                store.start(name, agent)
                agent.new_episode()
                play(local_env(Maze(seed=1, colours=colours)), agent.act, 500, agent)
                store.save(name, agent, 1)
            generic = {p['key'] for p in store.generic()}
            self.assertIn(('contact_goal', 'unique'), generic)
            self.assertIn(('direction', 1, (-1, 0)), generic)
            # a maze in other colours: once one move is seen, the other directions are taken from the store,
            # and the only object of its pattern (the exit) is tried as the goal
            agent = ObjectAgent(random.Random(0))
            store.start('c', agent)
            agent.new_episode()
            result = play(local_env(Maze(seed=2, colours=(2, 10, 11, 13, 15))), agent.act, 500, agent)
            self.assertEqual(result[0], 3)
            self.assertIn(('contact_goal', 'unique'), agent.priors_applied)
            self.assertTrue(any(k[0] == 'direction' for k in agent.priors_applied))
            self.assertEqual(agent.win_contacts, {((13,),)})
        # a generic direction that is wrong here is dropped once the action is seen to do otherwise
        agent = ObjectAgent(random.Random(0))
        agent.use_generic([{'key': ('direction', 1, (1, 0)), 'params': {}, 'context': [('acts_by', 'moves')]}])
        agent.new_episode()
        result = play(local_env(Maze(seed=2)), agent.act, 500, agent)
        self.assertEqual(result[0], 3)
        self.assertEqual(agent.directions.get(1, (-1, 0)), (-1, 0))


class TestMetaPatterns(unittest.TestCase):
    def test_elements_need_two_mechanisms(self):
        import tempfile
        from meta_patterns import MetaStore, arc_elements
        with tempfile.TemporaryDirectory() as d:
            m = MetaStore(d)
            m.add_game('tetris', {('drift', 1, (1, 0)): {}, ('transform', 1, 'turn right', (0, 0)): {}})
            self.assertEqual(m.elements(), {})          # each element seen one way only
            m.add_game('connect4', {('own_click', 'empty', 'drop', (1, 0)): {}})
            self.assertEqual(set(m.elements()), {'gravity'})     # a falling piece and a dropped disc
            m.add_arc('t1', 'C', 'rot90')
            self.assertIn('rotation', m.elements())     # a turning piece and an ARC rotation
            self.assertEqual(m.family_order(['A', 'C', 'K', 'N'])[:2], ['C', 'N'])
            self.assertEqual(MetaStore(d).elements().keys(), m.elements().keys())    # (kept in meta.json)
        self.assertEqual(arc_elements('K', 'rays from every element S in own colour, at obstacle stop'), {'alignment'})
        self.assertEqual(arc_elements('N', 'complete rot90 symmetry about the grid centre'), {'rotation', 'reflection'})
        # the repetition and cell-rule families
        self.assertEqual(arc_elements('R', 'repair the tiling under 0'), {'repetition'})
        self.assertEqual(arc_elements('S', 'cells of 5 recoloured by column parity from the tr corner'), {'repetition'})
        self.assertEqual(arc_elements('S', 'cells of 5 take the colour of the first other cell from the left'),
                         {'alignment'})
        with tempfile.TemporaryDirectory() as d:
            m = MetaStore(d)
            m.add_arc('t1', 'R', 'smallest repeated tile')
            m.add_arc('t2', 'E', 'tile 2x2')
            self.assertEqual(set(m.elements()), {'repetition'})     # two ARC families: a meta-pattern
            self.assertEqual(m.family_order(['A', 'C', 'E', 'R', 'S'])[:3], ['E', 'R', 'S'])

    @unittest.skipIf(ARCBaseGame is None, 'arcengine is not installed')
    def test_universal_elements_in_connect_four(self):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play
        # gravity and alignment as seen elsewhere (Tetris, ARC): no Connect Four or board game among them
        agent = ObjectAgent(random.Random(0))
        agent.use_meta({'gravity': {'params': [{'d': [1, 0], 'mechanism': 'drift'}, {'mechanism': 'arc:C'}]},
                        'alignment': {'params': [{'role': 'any', 'k': [16], 'full': True, 'mechanism': 'line_goal:any'},
                                                 {'mechanism': 'arc:K'}]}})
        agent.new_episode()
        play(local_env(GAMES['connect4'](seed=0)), agent.act, 100, agent, stop_on_loss=True)
        self.assertEqual(agent.meta_applied, {'gravity', 'alignment'})
        self.assertEqual(agent.click_models[0].kind(), 'drop')                  # the disc falls
        self.assertEqual(agent.others[0][0].kind(narrow=True), 'drop')         # the opponent's too


    @unittest.skipIf(ARCBaseGame is None, 'arcengine is not installed')
    def test_reach_and_shape_match_in_turn(self):
        from evaluate_arc3 import play
        for first in ('reach', 'shape_match'):
            agent = ObjectAgent(random.Random(0))
            agent.use_meta({'reach': {'params': [{'mechanism': 'contact_goal'}, {'mechanism': 'arc:F'}]},
                            'shape_match': {'params': [{'mechanism': 'shape_goal'}, {'mechanism': 'arc:O'}]}})
            if first == 'shape_match':
                agent._goal_turn = 'shape_match'            # as if shape_match had been tried first
            act, both = agent.act, []

            def checked(grid, actions):
                a = act(grid, actions)
                both.append(bool(agent.prior_contacts) and agent.prior_shape)
                return a
            agent.act = checked
            for _ in range(3):
                agent.new_episode()
                self.assertEqual(play(LocalEnv(merge_game()), agent.act, 1000, agent)[0], 1)
            self.assertFalse(any(both))                     # never both at once
            if first == 'reach':
                self.assertIn('reach', agent.meta_applied)
            else:
                self.assertNotIn('reach', agent.meta_applied)   # shape_match holds the turn: not used up

    def test_turn_passes_when_used_up(self):
        from collections import Counter
        agent = ObjectAgent(random.Random(0))
        agent.use_meta({'reach': {'params': []}, 'shape_match': {'params': []}})
        agent.rules[((8,),)] = Counter({'merge': 1})         # the agent has grown
        grid = np.zeros((5, 5), dtype=int)
        agent._goal_turn, agent._reach_used_up = 'reach', False
        agent._meta_usable(grid, {})
        self.assertFalse(agent.prior_shape)                  # reach still has objects to try
        agent._reach_used_up = True
        out = agent._meta_usable(grid, {})
        self.assertTrue(agent.prior_shape)                   # reach used up: shape_match's turn
        self.assertEqual(agent._goal_turn, 'shape_match')
        self.assertNotIn(('contact_goal', 'unique', 'inside'), out)
        agent.shape_rejected = True                          # a match reached without a win
        out = agent._meta_usable(grid, {})
        self.assertFalse(agent.prior_shape)
        self.assertEqual(agent._goal_turn, 'reach')
        self.assertIn(('contact_goal', 'unique', 'inside'), out)



@unittest.skipIf(ARCBaseGame is None, 'arcengine is not installed')
class TestCaptureGo(unittest.TestCase):
    def test_rules(self):
        from arc3_games import MINE, THEIRS, CaptureGo
        g = CaptureGo(seed=0)
        b = np.zeros((5, 5), dtype=np.int8)
        b[0, 0], b[0, 1] = THEIRS, MINE
        self.assertEqual(g._play(b.copy(), (1, 0), MINE), {(0, 0)})          # no liberty left: captured
        b = np.zeros((5, 5), dtype=np.int8)
        b[0, 1] = b[1, 0] = THEIRS
        self.assertIsNone(g._play(b.copy(), (0, 0), MINE))                  # no liberty, no capture: not allowed
        b[1, 1] = b[0, 2] = b[2, 0] = MINE
        self.assertEqual(g._play(b.copy(), (0, 0), MINE), {(0, 1), (1, 0)})  # ... unless it captures

    def test_game_ends_at_the_first_capture(self):
        from arc3_games import GAMES, local_env
        from evaluate_arc3 import play, random_choice
        results = set()
        for seed in range(6):
            env = local_env(GAMES['capture-go'](seed=seed))
            play(env, random_choice(random.Random(seed)), 2000, clicks=True, stop_on_loss=True)
            results.add(env.game.result)
        self.assertLessEqual(results, {'win', 'loss', 'draw', None})
        self.assertTrue(results & {'win', 'loss'})


    def test_group_effects_and_removal_goal(self):
        def board(stones):
            g = np.full((7, 7), 5)
            g[1:-1, 1:-1] = 0
            for p, v in stones.items():
                g[p] = v
            return g
        agent = ObjectAgent(random.Random(0))
        # a placement that removes nothing, then one that leaves a stone of 1 touching no empty cell: removed
        agent.observe(board({}), ('click', 3, 3), board({(3, 3): 2}))
        agent.observe(board({(1, 1): 1, (1, 2): 2}), ('click', 2, 1), board({(1, 2): 2, (2, 1): 2}))
        self.assertEqual(agent.group_rule['empty'], 0)
        self.assertEqual(agent.group_rule['cleared'], {1})
        self.assertIn(agent.click_models[0].kind(), ('map', 'offsets'))   # (it learned the placement alone)
        # a capture predicted elsewhere: the group of two 1s in the other corner
        g = board({(5, 4): 1, (5, 5): 1, (4, 4): 2, (4, 5): 2})
        after, _ = agent._click(g, None, (5, 3))
        self.assertTrue((after == board({(4, 4): 2, (4, 5): 2, (5, 3): 2})).all())
        # a win that removed cells of 1 makes removing cells of 1 the goal
        agent.level_won(g, ('click', 5, 3), frames=[after])
        self.assertEqual(agent.removal_goals, {1})
        self.assertTrue(agent.wins_with(g, None, ('click', 5, 3), agent._click(g, None, (5, 3))))


class TestFullGo(unittest.TestCase):
    def test_ko_and_area(self):
        from arc3_games import MINE, THEIRS, FullGo
        g = FullGo(seed=0)
        b = np.zeros((5, 5), dtype=np.int8)
        for p in ((0, 1), (1, 0), (2, 1)):
            b[p] = MINE
        for p in ((0, 2), (1, 1), (2, 2), (1, 3)):
            b[p] = THEIRS
        g.history = {b.tobytes()}
        self.assertEqual(g._play(b, (1, 2), MINE), {(1, 1)})
        g.history.add(b.tobytes())
        self.assertTrue(g._ko(b, (1, 1), THEIRS))           # taking back at once repeats the position
        self.assertNotIn((1, 1), g._allowed(b, THEIRS))
        b = np.zeros((5, 5), dtype=np.int8)
        b[0, :], b[1, :2], b[2, :] = MINE, MINE, THEIRS
        self.assertEqual((g.area(b, MINE), g.area(b, THEIRS)), (7, 15))   # (the region between borders both)

    def test_two_passes_end_the_game(self):
        from arcengine import GameAction
        from arc3_games import GAMES, local_env
        env = local_env(GAMES['go'](seed=0))
        env.reset()
        env.step(GameAction.ACTION5)            # the opponent, not behind, passes too: a draw on an empty board
        self.assertEqual(env.game.result, 'draw')

    def test_passes_and_count_goal(self):
        def board(stones):
            g = np.full((7, 7), 5)
            g[1:-1, 1:-1] = 0
            for p, v in stones.items():
                g[p] = v
            return g
        agent = ObjectAgent(random.Random(0))
        agent.observe(board({}), ('click', 3, 3), board({(3, 3): 2}))
        agent._learn_frames([board({(3, 3): 2}), board({(3, 3): 2, (1, 1): 1})])
        # a pass the opponent answers with a move: the game goes on
        g = board({(3, 3): 2, (1, 1): 1})
        g2 = board({(3, 3): 2, (1, 1): 1, (5, 5): 1})
        agent.after_step(g, 5, g2, frames=[g, g2])
        self.assertEqual(agent.pass_actions, {5})
        self.assertFalse(agent._count_known())
        # a round of passes that ends the game, won with more cells and more area
        ahead = board({(3, 3): 2, (3, 4): 2, (1, 1): 1})
        agent.level_won(ahead, 5, frames=[ahead, ahead])
        self.assertEqual(agent.count_goals, {'cells', 'area'})
        # one lost with more cells but less area: the count is by area
        stones = {(r, 2): 1 for r in range(1, 6)}
        stones.update({(r, 5): 2 for r in range(1, 6)})
        stones.update({(1, 3): 2, (1, 4): 2})
        behind = board(stones)
        self.assertEqual((agent._tally(behind, 'cells'), agent._tally(behind, 'area')), (2, -3))
        agent.lost(behind, 5, frames=[behind, behind])
        self.assertEqual(agent.count_goals, {'area'})
        self.assertTrue(agent._count_known())
        # ahead, after the opponent passed: passing wins
        agent._others_passed = True
        value, best = agent._search(ahead, [5, 6])
        self.assertEqual((value, best), (1.0, [5]))

    def test_placement_patterns(self):
        from shapes import canonical, symmetries
        def board(stones):
            g = np.full((7, 7), 5)
            g[1:-1, 1:-1] = 0
            for p, v in stones.items():
                g[p] = v
            return g
        p = ((1, 2, 3), (0, -1, 0), (0, 0, 3))
        self.assertTrue(all(canonical(q) == canonical(p) for q in symmetries(p)))
        agent = ObjectAgent(random.Random(0))
        agent.observe(board({}), ('click', 3, 3), board({(3, 3): 2}))
        agent._learn_frames([board({(3, 3): 2}), board({(3, 3): 2, (1, 1): 1})])
        # the agent places next to its stone, the opponent in a corner; the agent wins the game
        before = board({(3, 3): 2, (1, 1): 1})
        mine = board({(3, 3): 2, (3, 4): 2, (1, 1): 1})
        theirs = board({(3, 3): 2, (3, 4): 2, (1, 1): 1, (5, 5): 1})
        agent._record_moves(before, [mine, theirs])
        self.assertEqual([player for _, player in agent._game_moves], [0, 1])
        (k_mine, _), (k_theirs, _) = agent._game_moves
        agent._credit_moves(1.0)
        self.assertGreater(agent._pattern_value(k_mine), 0.5)        # placed by the winner: pursued
        self.assertLess(agent._pattern_value(k_theirs), 0.5)         # by the loser: avoided
        # the same placement turned a quarter: the same pattern
        self.assertEqual(agent._move_key(board({(3, 3): 2, (1, 1): 1}), (2, 3), agent._roles_for(0)),
                         agent._move_key(board({(3, 3): 2, (1, 1): 1}), (3, 4), agent._roles_for(0)))
        self.assertEqual(agent._placed_cell(before, mine, 0, agent._placements()), (3, 4))


if __name__ == '__main__':
    unittest.main()
