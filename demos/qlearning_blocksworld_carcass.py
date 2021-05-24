# System imports
import os
import sys
import copy

# 3rd-party library imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

# Framework imports
from mdp import BlocksWorldBuilder, VacuumCleanerWorldBuilder, SokobanBuilder
from mdp.abstraction import Carcass

from control import *
from policy import *


# Parameters
blocks_world_size = 7
episodes = 3000
max_episode_length = blocks_world_size * 3 + 1
learning_rate=0.03
epsilon=0.3

mdp_builder = BlocksWorldBuilder(blocks_world_size)

# Ground control
ground_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                     random_policy=RandomPolicy(),
                                                     qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                     epsilon=epsilon,
                                                     plan_for_new_states=False)
ground_target_policy = QTablePolicy(initial_value_estimate=0.0)
ground_control = QLearningControl(ground_target_policy, ground_behavior_policy, alpha=learning_rate)

# Abstract control
abstract_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                       random_policy=RandomPolicy(),
                                                       qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                       epsilon=epsilon,
                                                       plan_for_new_states=False)
abstract_target_policy = QTablePolicy(initial_value_estimate=0.0)
abstract_control = QLearningControl(abstract_target_policy, abstract_behavior_policy, alpha=learning_rate)


df = pd.DataFrame() # Dataframe for storing rewards, returns, etc.
adf = pd.DataFrame() # Dataframe for storing detailed information about the abstract policy

episode_ids = range(episodes)

for episode_id in episode_ids:

    print(f'\x1b[2K\rTraining:{episode_id * 100 / (episodes-1):3.0f}%', end='')

    mdp = mdp_builder.build_mdp()
    mdp_target = copy.deepcopy(mdp)
    amdp = Carcass(copy.deepcopy(mdp), 'blocksworld_stackordered.lp')
    amdp_target = copy.deepcopy(amdp)

    # Ground MDP
    ground_control.try_initialize_state(mdp.state, mdp.available_actions)
    ground_control.generate_episode_with_target_policy(mdp_target, step_limit=max_episode_length)
    ground_control.learn_episode(mdp, step_limit=max_episode_length)

    # Abstract MDP
    abstract_control.try_initialize_state(amdp.state, amdp.available_actions)
    abstract_control.generate_episode_with_target_policy(amdp_target, step_limit=max_episode_length)
    abstract_control.learn_episode(amdp, step_limit=max_episode_length)


    # Store all results in the dataframe
    row = {

        'episode_id': episode_id,
        'behavior_policy_return': mdp.return_history[0],
        'target_policy_return': mdp_target.return_history[0],
        'abstract_behavior_policy_return': amdp.return_history[0],
        'abstract_target_policy_return': amdp_target.return_history[0],
    }
    df = df.append(pd.Series(row, name=episode_id))

    arow = {
        ** { abstract_state: abstract_target_policy.suggest_action_for_state(abstract_state) 
                for abstract_state in abstract_target_policy._q_table.keys() }
    }
    adf = adf.append(pd.Series(arow, name=episode_id))

print()



_, (ax0,ax1,ax2) = plt.subplots(3,1, sharex=True)

ax0.plot(df['episode_id'], df['behavior_policy_return'], '.', alpha=0.7, label='Ground Behavior policy')
ax0.plot(df['episode_id'], df['target_policy_return'], '.', alpha=0.7, label='Ground Target policy')
ax0.plot(df['episode_id'], df['abstract_behavior_policy_return'], '.', alpha=0.7, label='Abstract Behavior policy')
ax0.plot(df['episode_id'], df['abstract_target_policy_return'], '.', alpha=0.7, label='Abstract Target policy')
ax0.set_xlabel('Episodes')
ax0.set_ylabel('Return')
ax0.set_title(f'{blocks_world_size}-blocks world')

attxt = f'''
{blocks_world_size}-blocks world
$\\alpha={learning_rate}$
$\\epsilon$-greedy exploration factor = {epsilon}
Maximum episode length = {max_episode_length}
'''
at = AnchoredText(attxt, loc='center left')
ax0.add_artist(at)
ax0.legend()

ax1.plot(df['episode_id'], df['behavior_policy_return'].cumsum(), label='Ground Behavior policy')
ax1.plot(df['episode_id'], df['target_policy_return'].cumsum(), label='Ground Target policy')
ax1.plot(df['episode_id'], df['abstract_behavior_policy_return'].cumsum(), label='Abstract Behavior policy')
ax1.plot(df['episode_id'], df['abstract_target_policy_return'].cumsum(), label='Abstract Target policy')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Cumulative Return')
ax1.legend()

abstract_states = list(reversed(sorted(adf.columns)))
for y, abstract_state in enumerate(abstract_states):
    best_actions = adf[abstract_state]
    all_best_actions = list(best_actions.unique())
    colors = plt.cm.get_cmap('Set1')(range(len(all_best_actions)))
    cmap = { a : c for a, c in zip(all_best_actions, colors) }

    bar_left = 0
    bar_width = 0
    bar_label = None
    for x, a in enumerate(best_actions):

        if a != bar_label:
            if bar_label:
                ax2.barh(y, bar_width, left=bar_left, height=0.4, color=cmap[bar_label])
            bar_left = x
            bar_width = 1
            bar_label = a

        else:
            bar_width += 1

    if bar_label:
        ax2.barh(y, bar_width, left=bar_left, height=0.4, color=cmap[bar_label])

    for x, label in enumerate(all_best_actions):
        ax2.text(x*(float(episodes) / len(all_best_actions)), y+0.25, label, color=cmap[label])

ax2.set_yticks(range(len(abstract_states)))
ax2.set_yticklabels(abstract_states)
ax2.set_ylim(-1,len(abstract_states))
ax2.set_xlabel('Episodes')

plt.show()
