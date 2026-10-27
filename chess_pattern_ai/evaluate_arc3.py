#!/usr/bin/env python3
"""
Play ARC-AGI-3 games with the effect learner (arc3_agent.py)

    python3 evaluate_arc3.py <environments_dir> [episodes] [max_actions] [seed] [agent] [game_id ...]

agent: 'cell' (Arc3Agent, the cell-level effect learner) or 'object'
(object_agent.ObjectAgent); default 'object'.

Games are loaded by the ARC-AGI toolkit in offline mode from
<environments_dir> (folders holding metadata.json and the game's .py file).
Each episode plays a fresh copy of the game, from level 1, for at most
max_actions actions (a lost level is reset; the reset is not counted); the
agent keeps what it learned across the episodes of one game. A random
player (uniform over the simple actions) gets the same budget.
Reported per episode: levels completed out of the game's levels, and the
actions each completed level took.

Clicks (ACTION6): an agent with `clicks = True` is offered action 6 when a
game lists it, and answers ('click', row, col) for a cell of the grid it
sees; the click is sent at that cell's centre pixel. The random player
picks an action id uniformly among those listed, and for 6 a random cell.

    python3 evaluate_arc3.py --click-games [episodes] [max_actions] [seed] [agent] [game ...]

plays the click games of arc3_click_games.py instead.
"""

import logging
import random
import sys

from arc3_agent import SIMPLE_ACTIONS, Arc3Agent, GridReader
from object_agent import ObjectAgent

AGENTS = {'cell': Arc3Agent, 'object': ObjectAgent}


def play(env, choose, max_actions: int, agent=None, clicks=None):
    """(levels completed, win_levels, actions per completed level, losses)"""
    from arcengine import GameAction, GameState
    fd = env.reset()
    read = GridReader()
    per_level, used, since, losses, best = [], 0, 0, 0, 0
    grid = read(fd.frame[-1])
    if clicks is None:
        clicks = getattr(agent, 'clicks', False)
    while used < max_actions and fd.state != GameState.WIN:
        if fd.state == GameState.GAME_OVER:
            losses += 1
            fd = env.step(GameAction.RESET)
            grid = read(fd.frame[-1])
            continue
        actions = [a for a in SIMPLE_ACTIONS if a in fd.available_actions]
        if clicks and 6 in fd.available_actions:
            actions.append(6)
        a = choose(grid, actions)
        if isinstance(a, tuple):
            x, y = read.pixel(a[1:])
            nxt = env.step(GameAction.ACTION6, {'x': x, 'y': y})
        else:
            nxt = env.step(GameAction.from_id(a))
        used += 1
        since += 1
        after = read(nxt.frame[-1])
        # (a game may end on its last level by WIN without counting the level as completed; a reset may
        # restart the game, so a level counts only the first time it is completed)
        if nxt.levels_completed > fd.levels_completed or nxt.state == GameState.WIN:
            # (and at most win_levels count: a game may count the last level again, as Complex Maze does
            # when its exit is reached as the energy runs out)
            if len(per_level) < nxt.win_levels and (nxt.levels_completed > best or nxt.state == GameState.WIN):
                best = max(best, nxt.levels_completed)
                per_level.append(since)
                since = 0
            if agent:
                agent.level_won(grid, a)
        elif nxt.state == GameState.GAME_OVER and agent:
            if getattr(agent, 'learns_from_losses', False):
                agent.after_step(grid, a, after)
            agent.lost(grid, a)
        elif agent:
            agent.after_step(grid, a, after)
        fd, grid = nxt, after
    return len(per_level), fd.win_levels, per_level, losses


def random_choice(rng):
    def choose(grid, actions):
        a = rng.choice(actions)
        return ('click', rng.randrange(grid.shape[0]), rng.randrange(grid.shape[1])) if a == 6 else a
    return choose


def main():
    logging.disable(logging.INFO)
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_actions = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    make = AGENTS[sys.argv[5]] if len(sys.argv) > 5 else ObjectAgent
    if sys.argv[1] == '--click-games':
        from arc3_click_games import GAMES, local_env
        names = sys.argv[6:] or sorted(GAMES)
        games = [(name, lambda name=name: local_env(GAMES[name]())) for name in names]
    else:
        from arc_agi import Arcade, OperationMode
        arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=sys.argv[1])
        names = sys.argv[6:] or sorted(e.game_id for e in arc.get_environments())
        games = [(gid, lambda gid=gid: arc.make(gid)) for gid in names]
    for gid, new_env in games:
        agent = make(random.Random(seed))
        rng = random.Random(seed)
        print(f'== {gid}')
        for ep in range(episodes):
            agent.new_episode()
            done, total, per_level, losses = play(new_env(), agent.act, max_actions, agent)
            print(f'   episode {ep + 1}: levels {done}/{total}, actions per level {per_level}, losses {losses}')
        done, total, per_level, losses = play(new_env(), random_choice(rng), max_actions, clicks=True)
        print(f'   random: levels {done}/{total}, actions per level {per_level}, losses {losses}')
        print(f'   agent: {agent.describe()}')


if __name__ == '__main__':
    main()
