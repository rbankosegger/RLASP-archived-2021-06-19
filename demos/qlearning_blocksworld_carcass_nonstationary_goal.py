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
blocks_world_size = 7
episodes = 3000
max_episode_length = blocks_world_size * 3 + 1
abstract_learning_rate = 0.05
ground_learning_rate = 0.3
epsilon = 0.2

blocks = [ f'b{i}' for i in range(blocks_world_size) ]
goals_1 = { f'subgoal({a},{b})' for a, b in zip(blocks, ['table'] + blocks) }
goals_2 = { f'subgoal({a},{b})' for a, b in zip(reversed(blocks), ['table'] + list(reversed(blocks))) }

mdp_builder_1 = BlocksWorldBuilder(blocks_world_size, state_static=goals_1)
mdp_builder_2 = BlocksWorldBuilder(blocks_world_size, state_static=goals_2)
switch_goal_at_episode = episodes / 2

# Ground control
ground_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                     random_policy=RandomPolicy(),
                                                     qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                     epsilon=epsilon,
                                                     plan_for_new_states=False)
ground_target_policy = QTablePolicy(initial_value_estimate=0.0)
ground_control = QLearningControl(ground_target_policy, ground_behavior_policy, alpha=ground_learning_rate)

# Abstract control
abstract_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                       random_policy=RandomPolicy(),
                                                       qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                       epsilon=epsilon,
                                                       plan_for_new_states=False)
abstract_target_policy = QTablePolicy(initial_value_estimate=0.0)
abstract_control = QLearningControl(abstract_target_policy, abstract_behavior_policy, alpha=abstract_learning_rate)

# Ground control with qtable wiped when goal changes
wiped_ground_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                     random_policy=RandomPolicy(),
                                                     qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                     epsilon=epsilon,
                                                     plan_for_new_states=False)
wiped_ground_target_policy = QTablePolicy(initial_value_estimate=0.0)
wiped_ground_control = QLearningControl(wiped_ground_target_policy, wiped_ground_behavior_policy, alpha=ground_learning_rate)

# Abstract control with qtable wiped when goal changes
wiped_abstract_behavior_policy = PlanningEpsilonGreedyPolicy(planner_policy=None,
                                                       random_policy=RandomPolicy(),
                                                       qtable_policy=QTablePolicy(initial_value_estimate=0.0),
                                                       epsilon=epsilon,
                                                       plan_for_new_states=False)
wiped_abstract_target_policy = QTablePolicy(initial_value_estimate=0.0)
wiped_abstract_control = QLearningControl(wiped_abstract_target_policy, wiped_abstract_behavior_policy, alpha=abstract_learning_rate)


df = pd.DataFrame() # Dataframe for storing rewards, returns, etc.
adf = pd.DataFrame() # Dataframe for storing detailed information about the abstract policy
wadf = pd.DataFrame() # Dataframe for storing detailed information about the abstract policy which has its qtable wiped


episode_ids = range(episodes)

for episode_id in episode_ids:

    print(f'\x1b[2K\rTraining:{episode_id * 100 / (episodes-1):3.0f}%', end='')

    if episode_id < switch_goal_at_episode:
        mdp_builder = mdp_builder_1
    else:
        mdp_builder = mdp_builder_2

    mdp = mdp_builder.build_mdp()
    mdp_target = copy.deepcopy(mdp)
    amdp = Carcass(copy.deepcopy(mdp), 'blocksworld_stackordered.lp')
    amdp_target = copy.deepcopy(amdp)
    wmdp = copy.deepcopy(mdp)
    wmdp_target = copy.deepcopy(mdp)
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

    if episode_id >= switch_goal_at_episode:

        # Ground MDP (wiped)
        wiped_ground_control.try_initialize_state(wmdp.state, wmdp.available_actions)
        wiped_ground_control.generate_episode_with_target_policy(wmdp_target, step_limit=max_episode_length)
        wiped_ground_control.learn_episode(wmdp, step_limit=max_episode_length)
    
        # Abstract MDP (wiped)
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

    if episode_id < switch_goal_at_episode:

        # For episodes before goals, make returns identical to the amdp
        row['wiped_behavior_policy_return'] = mdp.return_history[0]
        row['wiped_target_policy_return'] = mdp_target.return_history[0]
        row['wiped_abstract_behavior_policy_return'] = amdp.return_history[0]
        row['wiped_abstract_target_policy_return'] = amdp_target.return_history[0]

    else:

        # For episodes after switching goals, returns come from the "memory-wiped" agent
        row['wiped_behavior_policy_return'] = wmdp.return_history[0]
        row['wiped_target_policy_return'] = wmdp_target.return_history[0]
        row['wiped_abstract_behavior_policy_return'] = wamdp.return_history[0]
        row['wiped_abstract_target_policy_return'] = wamdp_target.return_history[0]
    
    df = df.append(pd.Series(row, name=episode_id))


    arow = { abstract_state: abstract_target_policy.suggest_action_for_state(abstract_state) 
                for abstract_state in abstract_target_policy._q_table.keys() }
    adf = adf.append(pd.Series(arow, name=episode_id))


    if episode_id >= switch_goal_at_episode:
        warow = { abstract_state: wiped_abstract_target_policy.suggest_action_for_state(abstract_state) 
                    for abstract_state in wiped_abstract_target_policy._q_table.keys() }
        wadf = wadf.append(pd.Series(warow, name=episode_id))

print()

_, (ax0,ax1,ax2,ax3) = plt.subplots(4,1, sharex=True)

#ax0.plot(df['episode_id'], df['behavior_policy_return'], 'o', alpha=0.7, label='Ground Behavior policy')
ax0.plot(df['episode_id'], df['target_policy_return'], 'o', alpha=0.7, label='Ground Target policy')
#ax0.plot(df['episode_id'], df['abstract_behavior_policy_return'], 'o', alpha=0.7, label='Abstract Behavior policy')
ax0.plot(df['episode_id'], df['abstract_target_policy_return'], 'o', alpha=0.7, label='Abstract Target policy')

#ax0.plot(df['episode_id'], df['wiped_behavior_policy_return'], 'o', alpha=0.7, label='Ground Behavior policy - QTable wiped')
ax0.plot(df['episode_id'], df['wiped_target_policy_return'], 'o', alpha=0.7, label='Ground Target policy - QTable wiped')
#ax0.plot(df['episode_id'], df['wiped_abstract_behavior_policy_return'], 'o', alpha=0.7, label='Abstract Behavior policy - QTable wiped')
ax0.plot(df['episode_id'], df['wiped_abstract_target_policy_return'], 'o', alpha=0.7, label='Abstract Target policy - QTable wiped')

ax0.set_xlabel('Episodes')
ax0.set_ylabel('Return')
ax0.set_title(f'{blocks_world_size}-blocks world, nonstationary goal')

attxt = f'''
Nonstationary {blocks_world_size}-blocks world. 
The goal is to stack in order until episode {switch_goal_at_episode}. 
Then, the goal is to stack in reversed order.

Abstract $\\alpha={abstract_learning_rate}$
Ground $\\alpha={ground_learning_rate}$
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

#ax1.plot(df['episode_id'], df['wiped_behavior_policy_return'].cumsum(), label='Ground Behavior policy - QTable wiped')
ax1.plot(df['episode_id'], df['wiped_target_policy_return'].cumsum(), label='Ground Target policy - QTable wiped')
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
        x += switch_goal_at_episode

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
ax3.set_title(f'Abstract target policy (qtable wiped at {switch_goal_at_episode} episodes).')

plt.show()
