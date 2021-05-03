from mdp import BlocksWorld, BlocksWorldBuilder
from mdp.abstraction import Carcass

bwb = BlocksWorldBuilder(blocks_world_size=5)

background_knowledge = """
    towerGreaterEqual(T, T, 1) :- on(T, table).
    towerGreaterEqual(T, B, N+1) :- towerGreaterEqual(T, A, N), on(B,A).
    towerHeight(T, N) :- N = #max { X : towerGreaterEqual(T, _, X) }, on(T,table).
    maxTowerHeight(N) :- N = #max { X : towerHeight(_, X) }.

    -correct(A) :- on(A,B), subgoal(A,C), B!=C.
    correctTowerRec(T, T, 1) :- on(T, table), not -correct(T).
    correctTowerRec(T, A, N+1) :- correctTowerRec(T, B, N), on(A,B), subgoal(A,B).
    correctTowerHeight(T, N) :- N = #max { 0; X : correctTowerRec(T, _, X) }, on(T,table).
    maxCorrectTowerHeight(N) :- N = #max { X : correctTowerHeight(_, X) }.

    -clear(A) :- on(B,A).
    clear(A) :- on(A,_), not -clear(A).

"""

rules = [
    
    """
    :- maxCorrectTowerHeight(N), N!=5.
    """,

    """
    :- maxCorrectTowerHeight(N), N!=4.
    """,

    """
    :- maxCorrectTowerHeight(N), N!=3.
    """,

    """
    :- maxCorrectTowerHeight(N), N!=2.
    """,

    """
    :- maxCorrectTowerHeight(N), N!=1.
    """,
]


abstract_states_collected = dict()


for state in bwb.all_states:

    mdp = BlocksWorld(state_initial=state, state_static={'subgoal(b0,b1)', 'subgoal(b1,b2)', 'subgoal(b2,b3)', 'subgoal(b3,b4)'})
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

