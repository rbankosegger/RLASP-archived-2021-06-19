import os
import sys
import unittest

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

# Framework imports
from mdp import BlocksWorld
from abstraction import Carcass

class TestAbstraction(unittest.TestCase):

    def test_abstraction(self):

        mdp = BlocksWorld(state_initial={'on(b1,b2)', 'on(b2,table)', 'on(b3,table)', 'on(b4,table)'}, 
                          state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})

        rules = [
            """
            abstractState1(A,B,C) :- on(A,B), on(B,table), on(C,table), A!=B, B!=C.
            abstractAction(move(x1,x3), move(A,C)) :- abstractState1(A,B,C).
            abstractAction(move(x3,x1), move(C,A)) :- abstractState1(A,B,C).
            abstractAction(move(x1,table), move(A,table)) :- abstractState1(A,B,C).
            :- not abstractState1(_,_,_).
            """
        ]

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('abstract0', abstract_mdp.state)
        self.assertEqual({'move(x1,x3)', 'move(x3,x1)', 'move(x1,table)'}, abstract_mdp.available_actions)

        self.assertEqual({'move(b1,b3)', 'move(b1,b4)'}, abstract_mdp.ground_actions_of('move(x1,x3)'))
        self.assertEqual({'move(b3,b1)', 'move(b4,b1)'}, abstract_mdp.ground_actions_of('move(x3,x1)'))
        self.assertEqual({'move(b1,table)'}, abstract_mdp.ground_actions_of('move(x1,table)'))

    def test_gutter(self):

        mdp = BlocksWorld(state_initial={'on(b1,b2)', 'on(b2,table)', 'on(b3,table)'}, 
                          state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})

        abstract_mdp = Carcass(mdp, rules=[])
        
        self.assertEqual('gutter', abstract_mdp.state)
        self.assertEqual({'gutter'}, abstract_mdp.available_actions)
        self.assertEqual(mdp.available_actions, abstract_mdp.ground_actions_of(abstract_action='gutter'))


    def test_three_blocksworld_example_alternative(self):

        # Implement and test the example 5.2.1 from Otterlo's PhD thesis (2008) p. 253
        # Try out alternative rule implementations as suggested in the text.

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

        # Test rule 1

        mdp = BlocksWorld(state_initial={'on(b1,b2)', 'on(b2,table)', 'on(b3,table)'}, 
                          state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('abstract0', abstract_mdp.state)
        self.assertEqual({'move(x1,x3)', 'move(x3,x1)', 'move(x1,table)'}, abstract_mdp.available_actions)

        self.assertEqual({'move(b1,b3)'}, abstract_mdp.ground_actions_of('move(x1,x3)'))
        self.assertEqual({'move(b3,b1)'}, abstract_mdp.ground_actions_of('move(x3,x1)'))
        self.assertEqual({'move(b1,table)'}, abstract_mdp.ground_actions_of('move(x1,table)'))


        # Test rule 2

        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,table)', 'on(b3,table)'}, 
                          state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('abstract1', abstract_mdp.state)
        self.assertEqual({'move(x1,x2)', 'move(x2,x1)', 'move(x1,x3)', 'move(x3,x1)', 'move(x2,x3)', 'move(x3,x2)'}, abstract_mdp.available_actions)

        # Observe: All moves have the same ground action set!!
        all_moves = {'move(b1,b2)','move(b2,b1)','move(b1,b3)','move(b3,b1)','move(b2,b3)','move(b3,b2)'}
        self.assertEqual(all_moves, abstract_mdp.ground_actions_of('move(x1,x2)'))
        self.assertEqual(all_moves, abstract_mdp.ground_actions_of('move(x2,x1)'))
        self.assertEqual(all_moves, abstract_mdp.ground_actions_of('move(x1,x3)'))
        self.assertEqual(all_moves, abstract_mdp.ground_actions_of('move(x3,x1)'))
        self.assertEqual(all_moves, abstract_mdp.ground_actions_of('move(x2,x3)'))
        self.assertEqual(all_moves, abstract_mdp.ground_actions_of('move(x3,x2)'))


        # Test rule 3
        mdp = BlocksWorld(state_initial={'on(b1,b2)', 'on(b2,b3)', 'on(b3,table)'}, 
                          state_static={'subgoal(b2,b1)', 'subgoal(b3,b2)'})

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('abstract2', abstract_mdp.state)
        self.assertEqual({'move(x1,table)'}, abstract_mdp.available_actions)

        self.assertEqual({'move(b1,table)'}, abstract_mdp.ground_actions_of('move(x1,table)'))


    def test_three_blocksworld_example_alternative(self):

        # Implement and test the example 5.2.1 from Otterlo's PhD thesis (2008) p. 253

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
            :- not abstractState2(_,_,_).
            """
        ]


        # Test alternative rule 2
        # Due to symmetrical substitutions, all 6 actions from the original rule have the same ground action set!
        # Instead, it is enough to define one action.

        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,table)', 'on(b3,table)'}, 
                          state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('abstract1', abstract_mdp.state)
        self.assertEqual({'move(x1,x2)'}, abstract_mdp.available_actions)

        # Observe: All moves have the same ground action set!!
        all_moves = {'move(b1,b2)','move(b2,b1)','move(b1,b3)','move(b3,b1)','move(b2,b3)','move(b3,b2)'}
        self.assertEqual(all_moves, abstract_mdp.ground_actions_of('move(x1,x2)'))



        # Test alternative rule 3 .
        # Rule 3 doesn't have to be explicitly defined. Without definition, every action related to rule 3 is automatically gathered in the gutter.
        mdp = BlocksWorld(state_initial={'on(b2,b1)', 'on(b1,b3)', 'on(b3,table)'}, 
                          state_static={'subgoal(b2,b1)', 'subgoal(b3,b2)'})

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('gutter', abstract_mdp.state)
        self.assertEqual({'gutter'}, abstract_mdp.available_actions)

        self.assertEqual({'move(b2,table)'}, abstract_mdp.ground_actions_of('gutter'))
