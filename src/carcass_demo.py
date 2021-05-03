from mdp import BlocksWorld, BlocksWorldBuilder
from mdp.abstraction import Carcass

bwb = BlocksWorldBuilder(blocks_world_size=3)

rules = [
    """
    abstractState1(A,B,C) :- on(A,B), on(B,table), on(C,table), A!=B, B!=C.
    abstractAction(move(x1,x3), move(A,C)) :- abstractState1(A,B,C).
    abstractAction(move(x3,x1), move(C,A)) :- abstractState1(A,B,C).
    abstractAction(move(x1,table), move(A,table)) :- abstractState1(A,B,C).
    :- not abstractState1(_,_,_).
    """,

    """
    abstractState2(A,B,C) :- on(A,table), on(B,table), on(C,table), A!=B, B!=C, A!=C.
    abstractAction(move(x1,x2), move(A,B)) :- abstractState2(A,B,C).
    abstractAction(move(x2,x1), move(B,A)) :- abstractState2(A,B,C).
    abstractAction(move(x1,x3), move(A,C)) :- abstractState2(A,B,C).
    abstractAction(move(x3,x1), move(C,A)) :- abstractState2(A,B,C).
    abstractAction(move(x2,x3), move(B,C)) :- abstractState2(A,B,C).
    abstractAction(move(x3,x2), move(C,B)) :- abstractState2(A,B,C).
    :- not abstractState2(_,_,_).
    """,

    """
    abstractState3(A,B,C) :- on(A,B), on(B,C), on(C,table), A!=B, B!=C, C!=table.
    abstractAction(move(x1,table), move(A,table)) :- abstractState3(A,B,C).
    :- not abstractState3(_,_,_).
    """
]


abstract_states_collected = dict()


for state in bwb.all_states:

    mdp = BlocksWorld(state_initial=state, state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})
    abstract_mdp = Carcass(mdp, rules)
    abstract_states_collected[abstract_mdp.state] = abstract_states_collected.get(abstract_mdp.state, set()) | {(mdp, abstract_mdp)}



print('+++')
print()
for abstract_state_label, ground_states in sorted(abstract_states_collected.items()):

    print('Abstract state:', abstract_state_label)
    
    for (mdp, abstract_mdp) in ground_states:
        print('\t', sorted(mdp.state))

        for abstract_action in abstract_mdp.available_actions:

            print('\t\t', 'ABSTRACT: ', abstract_action)
            print('\t\t', 'GROUND:   ', abstract_mdp.ground_actions_of(abstract_action))
        print()

    print()

