# System imports
import os
import sys
import copy

# 3rd-party library imports
import pandas as pd
from matplotlib import pyplot as plt

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
max_episode_length = blocks_world_size * 2 + 1

mdp_builder = BlocksWorldBuilder(blocks_world_size)

# Ground control
ground_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                     random_policy=RandomPolicy(),
                                                     qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                     epsilon=0.3,
                                                     plan_for_new_states=False)
ground_target_policy = QTablePolicy(initial_value_estimate=0.0)
ground_control = QLearningControl(ground_target_policy, ground_behavior_policy, alpha=0.1)

# Abstract control
abstract_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                       random_policy=RandomPolicy(),
                                                       qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                       epsilon=0.3,
                                                       plan_for_new_states=False)
abstract_target_policy = QTablePolicy(initial_value_estimate=0.0)
abstract_control = QLearningControl(abstract_target_policy, abstract_behavior_policy, alpha=0.1)


df = pd.DataFrame()

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


_, (ax0,ax1) = plt.subplots(2,1, sharex=True)

ax0.plot(df['episode_id'], df['behavior_policy_return'], label='Ground Behavior policy')
ax0.plot(df['episode_id'], df['target_policy_return'], label='Ground Target policy')
ax0.plot(df['episode_id'], df['abstract_behavior_policy_return'], label='Abstract Behavior policy')
ax0.plot(df['episode_id'], df['abstract_target_policy_return'], label='Abstract Target policy')
ax0.set_xlabel('Episodes')
ax0.set_ylabel('Return')
ax0.set_title(f'{blocks_world_size}-blocks world')
ax0.legend()

ax1.plot(df['episode_id'], df['behavior_policy_return'].cumsum(), label='Ground Behavior policy')
ax1.plot(df['episode_id'], df['target_policy_return'].cumsum(), label='Ground Target policy')
ax1.plot(df['episode_id'], df['abstract_behavior_policy_return'].cumsum(), label='Abstract Behavior policy')
ax1.plot(df['episode_id'], df['abstract_target_policy_return'].cumsum(), label='Abstract Target policy')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Cumulative Return')
ax1.legend()

plt.show()
