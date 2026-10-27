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

plays the small games of arc3_games.py instead (all but Tetris by
default); a game that takes a seed gets a different one in each episode
(and the random player another).

    python3 evaluate_arc3.py --score-games [episodes] [max_actions] [seed] [agent] [game ...]

plays games that count points as levels (Tetris by default): each episode
is one game, ending at the first loss; the score (levels_completed) is
reported.

Frames: an action can produce several frames (a game shows, for example,
the player's move and then each other player's). An agent with
`uses_frames = True` gets them all, as cell grids, in after_step,
level_won and lost; other agents get the last one.
"""

import logging
import random
import sys

from arc3_agent import SIMPLE_ACTIONS, Arc3Agent, GridReader
from object_agent import ObjectAgent

AGENTS = {'cell': Arc3Agent, 'object': ObjectAgent}


def play(env, choose, max_actions: int, agent=None, clicks=None, stop_on_loss=False):
    """
    (levels completed, win_levels, actions per completed level, losses, score): score is the
    highest levels_completed reached (for games that count points as levels). With stop_on_loss
    the episode ends at the first loss.
    """
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
            if stop_on_loss:
                break
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
        frames = [read(f) for f in nxt.frame] or [grid]
        after = frames[-1]
        extra = {'frames': frames} if getattr(agent, 'uses_frames', False) else {}
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
                agent.level_won(grid, a, **extra)
        elif nxt.state == GameState.GAME_OVER and agent:
            if getattr(agent, 'learns_from_losses', False):
                agent.after_step(grid, a, after, **extra)
            agent.lost(grid, a, **extra)
        elif agent:
            agent.after_step(grid, a, after, **extra)
        fd, grid = nxt, after
    return len(per_level), fd.win_levels, per_level, losses, best


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
    stop_on_loss = sys.argv[1] == '--score-games'
    if sys.argv[1] in ('--click-games', '--score-games'):
        from arc3_games import GAMES, local_env
        names = sys.argv[6:] or (['tetris'] if stop_on_loss else sorted(set(GAMES) - {'tetris'}))
        import inspect

        def maker(name):
            takes_seed = 'seed' in inspect.signature(GAMES[name]).parameters
            count = [0]

            def new_env():
                count[0] += 1
                return local_env(GAMES[name](seed=seed * 1000 + count[0]) if takes_seed else GAMES[name]())
            return new_env
        games = [(name, maker(name)) for name in names]
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
            done, total, per_level, losses, score = play(new_env(), agent.act, max_actions, agent,
                                                         stop_on_loss=stop_on_loss)
            print(f'   episode {ep + 1}: levels {done}/{total}, actions per level {per_level}, losses {losses}'
                  + (f', score {score}' if stop_on_loss else ''))
        done, total, per_level, losses, score = play(new_env(), random_choice(rng), max_actions, clicks=True,
                                                     stop_on_loss=stop_on_loss)
        print(f'   random: levels {done}/{total}, actions per level {per_level}, losses {losses}'
              + (f', score {score}' if stop_on_loss else ''))
        print(f'   agent: {agent.describe()}')


if __name__ == '__main__':
    main()
