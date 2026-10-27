#!/usr/bin/env python3
"""Unit tests for arc3_agent and evaluate_arc3 (run: python3 -m unittest test_arc3_agent)"""

import random
import unittest

import numpy as np

from arc3_agent import Arc3Agent, GridReader, cell_grid, find_scale

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


class LocalEnv:
    """The reset/step interface of the toolkit's environment wrappers, over a game object"""

    def __init__(self, game):
        self.game = game

    def reset(self):
        return self.game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

    def step(self, action):
        return self.game.perform_action(ActionInput(id=action), raw=True)


class TestGrid(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
