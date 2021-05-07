import os
import sys

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import random

from mdp import BlocksWorld, BlocksWorldBuilder
from mdp.abstraction import Carcass


bwb = BlocksWorldBuilder(blocks_world_size=5)
state_static = set(f'subgoal({x},{y})' for (x,y) in zip(bwb.block_terms, ['table']+bwb.block_terms))
# state_static = {'subgoal(b3,b2)', 'subgoal(b4,b3)', 'subgoal(b1,b0)', 'subgoal(b2,b1)'}


abstract_states_collected = dict()

num_5blocksworld_states=501

for state in bwb.all_states[:num_5blocksworld_states]:

    mdp = BlocksWorld(state, state_static)
    abstract_mdp = Carcass(mdp, 'blocksworld_stackordered.lp')
    abstract_states_collected[abstract_mdp.state] = abstract_states_collected.get(abstract_mdp.state, set()) | {(mdp, abstract_mdp)}



print('+++')
print('Goal:', state_static)
print()
for abstract_state_label, ground_states in sorted(abstract_states_collected.items()):

    print('Abstract state:', abstract_state_label, f'({len(ground_states)} states)')

    for (mdp, abstract_mdp) in random.sample(ground_states, min(len(ground_states), 5)):
        print('\t', sorted(mdp.state))

        for abstract_action in abstract_mdp.available_actions:

            print('\t\t', 'ABSTRACT: ', abstract_action)
            print('\t\t', 'GROUND:   ', abstract_mdp.ground_actions_of(abstract_action))
        print()

    print()

