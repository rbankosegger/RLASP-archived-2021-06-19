# System imports
import os
import sys
import copy
from datetime import datetime

# 3rd-party library imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.dates import SecondLocator, MicrosecondLocator

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

# Framework imports
from mdp import BlocksWorldBuilder, VacuumCleanerWorldBuilder, SokobanBuilder
from control import *
from policy import *


# Parameters
blocks_world_size = 5
episodes = 3000
max_episode_length = blocks_world_size * 3 + 1 
planning_horizon = blocks_world_size + 1
learning_rate=0.3
epsilon=0.2

mdp_builder = BlocksWorldBuilder(blocks_world_size)
pre_generated_mdps = [ mdp_builder.build_mdp() for _ in range(episodes) ]

class RealTimeRewardHistory:

    def __init__(self):
        self.df = pd.DataFrame()

    def callback(self, next_reward, **kwargs):

        row = {
            'reward': next_reward,
            'time': datetime.now(),
        }


        self.df = self.df.append(pd.Series(row), ignore_index=True)

callback = RealTimeRewardHistory()
planning_callback = RealTimeRewardHistory()



# Create an epsilon greedy policy with no planning
behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                              random_policy=RandomPolicy(),
                                              qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                              epsilon=epsilon,
                                              plan_for_new_states=False)

target_policy = QTablePolicy(initial_value_estimate=0.0)

control = QLearningControl(target_policy, behavior_policy, alpha=learning_rate)

# Create an epsilon greedy policy with planning
planning_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=PlannerPolicy(planning_horizon, mdp_builder),
                                              random_policy=RandomPolicy(),
                                              qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                              epsilon=epsilon,
                                              plan_for_new_states=True)

planning_target_policy = QTablePolicy(initial_value_estimate=0.0)

planning_control = QLearningControl(planning_target_policy, planning_behavior_policy, alpha=learning_rate)


df = pd.DataFrame()

pre_generated_mdps = [ mdp_builder.build_mdp() for _ in range(episodes) ]


# Run episodes for non-planner
print('Start training for tabula rasa agent')
for episode_id, original_mdp in enumerate(pre_generated_mdps):

    print(f'\x1b[2K\rTraining:{episode_id * 100 / (episodes-1):3.0f}%', end='')
    
    mdp = copy.deepcopy(original_mdp)

    control.try_initialize_state(mdp.state, mdp.available_actions)
    control.learn_episode(mdp, step_limit=max_episode_length, per_step_callback=callback)
print()

# Run episodes for non-planner
print('Start training for planning agent')
for episode_id, original_mdp in enumerate(pre_generated_mdps):

    print(f'\x1b[2K\rTraining:{episode_id * 100 / (episodes-1):3.0f}%', end='')
    
    mdp = copy.deepcopy(original_mdp)

    planning_control.try_initialize_state(mdp.state, mdp.available_actions)
    planning_control.learn_episode(mdp, step_limit=max_episode_length, per_step_callback=planning_callback)
print()


df = callback.df
df['rel_time_seconds'] = (df['time'] - df['time'].iloc[0]) / np.timedelta64(1, 's')
df['delta_time_seconds'] = df['time'].diff() / np.timedelta64(1, 's')

pdf = planning_callback.df
pdf['rel_time_seconds'] = (pdf['time'] - pdf['time'].iloc[0]) / np.timedelta64(1, 's')
pdf['delta_time_seconds'] = pdf['time'].diff() / np.timedelta64(1, 's')


_, (ax0,ax1,ax2) = plt.subplots(3,1, sharex=False)

ax0.plot(df['rel_time_seconds'], df['reward'], '.', label='Behavior policy (no planning)', alpha=0.5)
ax0.plot(pdf['rel_time_seconds'], pdf['reward'], '.', label='Behavior policy (with planning)', alpha=0.5)
ax0.set_xlabel('Time since start of training (s)')
ax0.set_ylabel('Reward per timestep')
#ax0.xaxis.set_major_locator(SecondLocator())
#ax0.xaxis.set_minor_locator(SecondLocator())

attxt = f'''
{blocks_world_size}-blocks world
Episodes = {episodes}
$\\alpha={learning_rate}$
$\\epsilon$-greedy exploration factor = {epsilon}
Maximum episode length = {max_episode_length}
Planning horizon = {planning_horizon}
'''
at = AnchoredText(attxt, loc='center left')
ax0.add_artist(at)
ax0.legend(loc='center right')

ax1.plot(df['rel_time_seconds'], df['reward'].cumsum(), '-', label='Behavior policy (no planning)')
ax1.plot(pdf['rel_time_seconds'], pdf['reward'].cumsum(), '-', label='Behavior policy (with planning)')
ax1.set_xlabel('Time since start of training (s)')
ax1.set_ylabel('Cumulative reward per timestep')
ax1.legend()

ax2.plot(range(len(df)-1), 1000 * df['delta_time_seconds'].iloc[1:], '.', label='Behavior policy (no planning)', alpha=0.5)
ax2.plot(range(len(pdf)-1), 1000 * pdf['delta_time_seconds'].iloc[1:], '.', label='Behavior policy (with planning)', alpha=0.5)
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Duration of timestep (ms)')
ax2.legend()

plt.suptitle(f'Planning in a {blocks_world_size}-blocks world')

plt.show()
