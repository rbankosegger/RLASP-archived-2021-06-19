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
from control import *
from policy import *


# Parameters
blocks_world_size = 4
episodes = 100
max_episode_length = 9

mdp_builder = BlocksWorldBuilder(blocks_world_size)

# Create an epsilon greedy policy with no planning
behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                              random_policy=RandomPolicy(),
                                              qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                              epsilon=0.3,
                                              plan_for_new_states=False)

target_policy = QTablePolicy(initial_value_estimate=0.0)

control = QLearningControl(target_policy, behavior_policy, alpha=0.3)


df = pd.DataFrame()

episode_ids = range(episodes)

for episode_id in episode_ids:

    print(f'\x1b[2K\rTraining:{episode_id * 100 / (episodes-1):3.0f}%', end='')
    
    mdp = mdp_builder.build_mdp()
    control.try_initialize_state(mdp.state, mdp.available_actions)

    # First, test the target policy and see how it would perform 
    mdp_target = copy.deepcopy(mdp)
    control.generate_episode_with_target_policy(mdp_target, step_limit=max_episode_length)

    # Second, train the target policy and the behavior policy on the mdp
    control.learn_episode(mdp, step_limit=max_episode_length)

    # Finally, store all results in the dataframe
    row = {

        'episode_id': episode_id,
        'behavior_policy_return': mdp.return_history[0],
        'target_policy_return': mdp_target.return_history[0],
    }
    
    df = df.append(pd.Series(row, name=episode_id))


_, (ax0,ax1) = plt.subplots(2,1, sharex=True)

ax0.plot(df['episode_id'], df['behavior_policy_return'], label='Behavior policy')
ax0.plot(df['episode_id'], df['target_policy_return'], label='Target policy')
ax0.set_xlabel('Episodes')
ax0.set_ylabel('Return')
ax0.legend()

ax1.plot(df['episode_id'], df['behavior_policy_return'].cumsum(), label='Behavior policy')
ax1.plot(df['episode_id'], df['target_policy_return'].cumsum(), label='Target policy')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Cumulative Return')
ax1.legend()

plt.show()
