# System imports
import os
import sys
import copy

# 3rd-party library imports
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
blocks_world_size_1 = 4
blocks_world_size_2 = 7
episodes = 300
switch_size_at_episode = 100
max_episode_length = max(blocks_world_size_1, blocks_world_size_2) * 3 + 1
learning_rate=0.1
epsilon=0.2

mdp_builder_1 = BlocksWorldBuilder(blocks_world_size_1)
mdp_builder_2 = BlocksWorldBuilder(blocks_world_size_2)

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

# Abstract control with qtable wiped when blocks world size changes
wiped_abstract_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                       random_policy=RandomPolicy(),
                                                       qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                       epsilon=epsilon,
                                                       plan_for_new_states=False)
wiped_abstract_target_policy = QTablePolicy(initial_value_estimate=0.0)
wiped_abstract_control = QLearningControl(wiped_abstract_target_policy, wiped_abstract_behavior_policy, alpha=learning_rate)


df = pd.DataFrame() # Dataframe for storing rewards, returns, etc.
adf = pd.DataFrame() # Dataframe for storing detailed information about the abstract policy
wadf = pd.DataFrame() # Dataframe for storing detailed information about the abstract policy which has its qtable wiped

episode_ids = range(episodes)

for episode_id in episode_ids:

    if episode_id < switch_size_at_episode:
        current_mdp_builder = mdp_builder_1
    else:
        current_mdp_builder = mdp_builder_2

    print(f'\x1b[2K\rTraining:{episode_id * 100 / (episodes-1):3.0f}%', end='')

    mdp = current_mdp_builder.build_mdp()
    mdp_target = copy.deepcopy(mdp)
    amdp = Carcass(copy.deepcopy(mdp), 'blocksworld_stackordered.lp')
    amdp_target = copy.deepcopy(amdp)
    wamdp = copy.deepcopy(amdp)
    wamdp_target = copy.deepcopy(amdp)

    # Ground MDP
    ground_control.try_initialize_state(mdp.state, mdp.available_actions)
    ground_control.generate_episode_with_target_policy(mdp_target, step_limit=max_episode_length)
    ground_control.learn_episode(mdp, step_limit=max_episode_length)

    # Abstract MDP
    abstract_control.try_initialize_state(amdp.state, amdp.available_actions)
    abstract_control.generate_episode_with_target_policy(amdp_target, step_limit=max_episode_length)
    abstract_control.learn_episode(amdp, step_limit=max_episode_length)

    # Abstract MDP with memory wiped when block size changes
    if episode_id >= switch_size_at_episode:

        wiped_abstract_control.try_initialize_state(wamdp.state, wamdp.available_actions)
        wiped_abstract_control.generate_episode_with_target_policy(wamdp_target, step_limit=max_episode_length)
        wiped_abstract_control.learn_episode(wamdp, step_limit=max_episode_length)


    # Store all results in the dataframe
    row = {

        'episode_id': episode_id,
        'behavior_policy_return': mdp.return_history[0],
        'target_policy_return': mdp_target.return_history[0],
        'abstract_behavior_policy_return': amdp.return_history[0],
        'abstract_target_policy_return': amdp_target.return_history[0],
    }

    if episode_id < switch_size_at_episode:

        # For episodes before switching block size, make returns identical to the amdp
        row['wiped_abstract_behavior_policy_return'] = amdp.return_history[0]
        row['wiped_abstract_target_policy_return'] = amdp_target.return_history[0]

    else:

        # For episodes after switching block size, returns come from the "memory-wiped" agent
        row['wiped_abstract_behavior_policy_return'] = wamdp.return_history[0]
        row['wiped_abstract_target_policy_return'] = wamdp_target.return_history[0]

    df = df.append(pd.Series(row, name=episode_id))


    arow = {
        ** { abstract_state: abstract_target_policy.suggest_action_for_state(abstract_state) 
                for abstract_state in abstract_target_policy._q_table.keys() }
    }
    adf = adf.append(pd.Series(arow, name=episode_id))


    if episode_id >= switch_size_at_episode:
        warow = {
            ** { abstract_state: wiped_abstract_target_policy.suggest_action_for_state(abstract_state) 
                    for abstract_state in wiped_abstract_target_policy._q_table.keys() }
        }
        wadf = wadf.append(pd.Series(warow, name=episode_id))

print()


_, (ax0,ax1,ax2,ax3) = plt.subplots(4,1, sharex=True)

#ax0.plot(df['episode_id'], df['behavior_policy_return'], 'o', label='Ground Behavior policy', alpha=.7)
ax0.plot(df['episode_id'], df['target_policy_return'], 'o', label='Ground Target policy', alpha=.7)
#ax0.plot(df['episode_id'], df['abstract_behavior_policy_return'], 'o', label='Abstract Behavior policy', alpha=.7)
ax0.plot(df['episode_id'], df['abstract_target_policy_return'], 'o', label='Abstract Target policy', alpha=.7)
#ax0.plot(df['episode_id'], df['wiped_abstract_behavior_policy_return'], 'o', label='Abstract Behavior policy - QTable wiped', alpha=.7)
ax0.plot(df['episode_id'], df['wiped_abstract_target_policy_return'], 'o', label='Abstract Target policy - QTable wiped', alpha=.7)
ax0.set_xlabel('Episodes')
ax0.set_ylabel('Return')
ax0.set_title(f'Nonstationary {blocks_world_size_1}/{blocks_world_size_2}-blocks world')

attxt = f'''
Nonstationary blocks world. 
{blocks_world_size_1} blocks until episode {switch_size_at_episode}, then {blocks_world_size_2}.
$\\alpha={learning_rate}$
$\\epsilon$-greedy exploration factor = {epsilon}
Maximum episode length = {max_episode_length}
'''
at = AnchoredText(attxt, loc='center left')
ax0.add_artist(at)
ax0.legend(loc='center right')

#ax1.plot(df['episode_id'], df['behavior_policy_return'].cumsum(), label='Ground Behavior policy')
ax1.plot(df['episode_id'], df['target_policy_return'].cumsum(), label='Ground Target policy')
#ax1.plot(df['episode_id'], df['abstract_behavior_policy_return'].cumsum(), label='Abstract Behavior policy')
ax1.plot(df['episode_id'], df['abstract_target_policy_return'].cumsum(), label='Abstract Target policy')
#ax1.plot(df['episode_id'], df['wiped_abstract_behavior_policy_return'].cumsum(), label='Abstract Behavior policy - QTable wiped')
ax1.plot(df['episode_id'], df['wiped_abstract_target_policy_return'].cumsum(), label='Abstract Target policy - QTable wiped')
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
ax2.set_title('Abstract target policy')

abstract_states = list(reversed(sorted(wadf.columns)))
for y, abstract_state in enumerate(abstract_states):

    best_actions = wadf[abstract_state]
    all_best_actions = list(best_actions.unique())
    colors = plt.cm.get_cmap('Set1')(range(len(all_best_actions)))
    cmap = { a : c for a, c in zip(all_best_actions, colors) }

    bar_left = 0
    bar_width = 0
    bar_label = None
    for x, a in enumerate(best_actions):

        # Wiped qtable starts only after episode size is switched
        x += switch_size_at_episode

        if a != bar_label:
            if bar_label:
                ax3.barh(y, bar_width, left=bar_left, height=0.4, color=cmap[bar_label])
            bar_left = x
            bar_width = 1
            bar_label = a

        else:
            bar_width += 1

    if bar_label:
        ax3.barh(y, bar_width, left=bar_left, height=0.4, color=cmap[bar_label])

    for x, label in enumerate(all_best_actions):
        ax3.text(x*(float(episodes) / len(all_best_actions)), y+0.25, label, color=cmap[label])

ax3.set_yticks(range(len(abstract_states)))
ax3.set_yticklabels(abstract_states)
ax3.set_ylim(-1,len(abstract_states))
ax3.set_xlabel('Episodes')
ax3.set_title(f'Abstract target policy (qtable wiped at {switch_size_at_episode} episodes).')

plt.show()
