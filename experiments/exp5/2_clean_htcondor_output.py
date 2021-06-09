import glob
import pandas as pd
from tqdm import tqdm

print('Loading files...')
gs = []
for file in tqdm(glob.glob('htcondor_workspace/*.csv')):

    g = pd.read_csv(file)

    # Compute cumulative returns for all runs
    g['behavior_policy_return_cumulative'] = g['behavior_policy_return'].cumsum()
    g['target_policy_return_cumulative'] = g['target_policy_return'].cumsum()

    gs.append(g)

print('Concatenating...')
df = pd.concat(gs, ignore_index=True)

print('Misc processing...')
df['arg_learning_rate'] = df['arg_learning_rate'].fillna('-')

# Make episodes start at 1
df.episode_id += 1

def mdp_label(row):
    if row.arg_mdp == 'blocksworld':
        return f'{row.arg_blocks_world_size:1.0f}-Blocks World'
    if row.arg_mdp == 'sokoban':
        return f'Sokoban ({row.arg_sokoban_level_name})'

df['mdp_label'] = df.apply(mdp_label, axis=1)

def algorithm_label(row):
    if row.arg_control_algorithm == 'q_learning':
        return 'Standard QL'
    if row.arg_control_algorithm == 'q_learning_reversed_update':
        return 'Reversed-update QL'
    if row.arg_control_algorithm == 'monte_carlo':
        return 'First-visit MC'

df['algorithm_label'] = df.apply(algorithm_label, axis=1)

def planner_label(row):
    if row.arg_plan_for_new_states:
        return f'Plan on first visit (ph={row.arg_planning_horizon})'
    else:
        return 'Tabula rasa'

df['planner_label'] = df.apply(planner_label, axis=1)

def full_algorithm_label(row):

    if row.arg_control_algorithm == 'monte_carlo':
        return row.algorithm_label
    else:
        return f'{row.algorithm_label}, $\\alpha={row.arg_learning_rate}$'

df['full_algorithm_label'] = df.apply(full_algorithm_label, axis=1)

print('Saving...')
df.to_csv('2_experiment_results.csv')

