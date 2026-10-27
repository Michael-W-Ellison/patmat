#!/usr/bin/env python3
"""
Play ARC-AGI-3 games with the effect learner (arc3_agent.py)

    python3 evaluate_arc3.py <environments_dir> [episodes] [max_actions] [seed] [game_id ...]

Games are loaded by the ARC-AGI toolkit in offline mode from
<environments_dir> (folders holding metadata.json and the game's .py file).
Each episode plays a fresh copy of the game, from level 1, for at most
max_actions actions (a lost level is reset; the reset is not counted); the
agent keeps what it learned across the episodes of one game. A random
player (uniform over the simple actions) gets the same budget.
Reported per episode: levels completed out of the game's levels, and the
actions each completed level took.
"""

import logging
import random
import sys

from arc3_agent import SIMPLE_ACTIONS, Arc3Agent, GridReader


def play(env, choose, max_actions: int, agent=None):
    """(levels completed, win_levels, actions per completed level, losses)"""
    from arcengine import GameAction, GameState
    fd = env.reset()
    read = GridReader()
    per_level, used, since, losses = [], 0, 0, 0
    grid = read(fd.frame[-1])
    while used < max_actions and fd.state != GameState.WIN:
        if fd.state == GameState.GAME_OVER:
            losses += 1
            if agent:
                agent.lost()
            fd = env.step(GameAction.RESET)
            grid = read(fd.frame[-1])
            continue
        actions = [a for a in SIMPLE_ACTIONS if a in fd.available_actions]
        a = choose(grid, actions)
        nxt = env.step(GameAction.from_id(a))
        used += 1
        since += 1
        after = read(nxt.frame[-1])
        # (a game may end on its last level by WIN without counting the level as completed)
        if nxt.levels_completed > fd.levels_completed or nxt.state == GameState.WIN:
            per_level.append(since)
            since = 0
            if agent:
                agent.level_won(grid, a)
        elif nxt.state != GameState.GAME_OVER and agent:
            agent.after_step(grid, a, after)
        fd, grid = nxt, after
    return len(per_level), fd.win_levels, per_level, losses


def main():
    logging.disable(logging.INFO)
    from arc_agi import Arcade, OperationMode
    env_dir = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_actions = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    games = sys.argv[5:] or sorted(e.game_id for e in arc.get_environments())
    for gid in games:
        agent = Arc3Agent(random.Random(seed))
        rng = random.Random(seed)
        print(f'== {gid}')
        for ep in range(episodes):
            agent.new_episode()
            done, total, per_level, losses = play(arc.make(gid), agent.act, max_actions, agent)
            print(f'   episode {ep + 1}: levels {done}/{total}, actions per level {per_level}, losses {losses}')
        done, total, per_level, losses = play(arc.make(gid), lambda g, acts: rng.choice(acts), max_actions)
        print(f'   random: levels {done}/{total}, actions per level {per_level}, losses {losses}')
        print(f'   agent: values {agent.agent_values}, directions {agent.directions}, '
              f'model {getattr(agent.model(), "name", None)}, winning moves {sorted(agent.win_keys)}')


if __name__ == '__main__':
    main()
