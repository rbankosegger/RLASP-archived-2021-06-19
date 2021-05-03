from mdp import BlocksWorld, BlocksWorldBuilder
from mdp.abstraction import Carcass

bwb = BlocksWorldBuilder(blocks_world_size=5)

background_knowledge = """
    towerGreaterEqual(T, T, 1) :- on(T, table).
    towerGreaterEqual(T, B, N+1) :- towerGreaterEqual(T, A, N), on(B,A).
    towerHeight(T,N) :- N = #max { X : towerGreaterEqual(T, _, X) }, on(T,table).
    maxTowerHeight(N) :- N = #max { X : towerHeight(_, X) }.

"""

rules = [
    """
    :- maxTowerHeight(N), N != 5.
    """,

    """
    :- maxTowerHeight(N), N != 4.
    """,

    """
    :- maxTowerHeight(N), N != 3.
    """,
    
    """
    :- maxTowerHeight(N), N != 2.
    """,
]


abstract_states_collected = dict()


for state in bwb.all_states:

    mdp = BlocksWorld(state_initial=state, state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})
    abstract_mdp = Carcass(mdp, rules, background_knowledge)
    abstract_states_collected[abstract_mdp.state] = abstract_states_collected.get(abstract_mdp.state, set()) | {(mdp, abstract_mdp)}



print('+++')
print()
for abstract_state_label, ground_states in sorted(abstract_states_collected.items()):

    print('Abstract state:', abstract_state_label, f'({len(ground_states)} states)')

    for (mdp, abstract_mdp) in list(ground_states)[:3]:
        print('\t', sorted(mdp.state))

        for abstract_action in abstract_mdp.available_actions:

            print('\t\t', 'ABSTRACT: ', abstract_action)
            print('\t\t', 'GROUND:   ', abstract_mdp.ground_actions_of(abstract_action))
        print()

    print()

