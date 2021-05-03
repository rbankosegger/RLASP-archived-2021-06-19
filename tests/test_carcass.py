import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

# Framework imports
from mdp import BlocksWorld
from mdp.abstraction import Carcass

class TestAbstraction(unittest.TestCase):


    def test_gutter(self):

        # All states that are not matched by a rule end up in a "gutter" state with only one (random) available action.
        # In this case, no rules were specified, thus all states end up in the gutter.

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


    def test_state_transition(self):

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

        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,b1)', 'on(b3,table)'}, 
                          state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('abstract0', abstract_mdp.state)

        next_state, next_reward = abstract_mdp.transition('move(x1,x3)')
        self.assertEqual('abstract0', next_state)
        self.assertEqual(-1, next_reward)
        self.assertEqual('abstract0', abstract_mdp.state)
        self.assertEqual({'on(b1,table)', 'on(b2,b3)', 'on(b3,table)'}, mdp.state)

        next_state, next_reward = abstract_mdp.transition('move(x3,x1)')
        self.assertEqual('gutter', next_state)
        self.assertEqual(100-1, next_reward)
        self.assertEqual('gutter', abstract_mdp.state)
        self.assertEqual({'on(b3,table)', 'on(b2,b3)', 'on(b1,b2)'}, mdp.state)

        # Check if trajectory is correct: S0, A0, R1, S1, A1, R2, S2
        self.assertEqual('abstract0', abstract_mdp.state_history[0]) # S0
        self.assertEqual('move(x1,x3)', abstract_mdp.action_history[0]) # A0
        self.assertEqual(-1, abstract_mdp.reward_history[1]) # R1
        self.assertEqual('abstract0', abstract_mdp.state_history[1]) #S1
        self.assertEqual('move(x3,x1)', abstract_mdp.action_history[1]) # A1
        self.assertEqual(100-1, abstract_mdp.reward_history[2]) # R2
        self.assertEqual('gutter', abstract_mdp.state_history[2]) #S2

        # Test returns
        self.assertEqual(-1 + 99, abstract_mdp.return_history[0])
        self.assertEqual(99, abstract_mdp.return_history[1])
    
    def test_state_transition_multiple_ground_actions(self):

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

        mdp = BlocksWorld(state_initial={'on(b1,b2)', 'on(b2,table)', 'on(b3,table)', 'on(b4,table)'}, 
                          state_static={'subgoal(b1,b2)', 'subgoal(b2,b3)'})

        abstract_mdp = Carcass(mdp, rules)

        self.assertEqual('abstract0', abstract_mdp.state)
        self.assertEqual({'move(x1,x3)', 'move(x3,x1)', 'move(x1,table)'}, abstract_mdp.available_actions)

        self.assertEqual({'move(b1,b3)', 'move(b1,b4)'}, abstract_mdp.ground_actions_of('move(x1,x3)'))
        self.assertEqual({'move(b3,b1)', 'move(b4,b1)'}, abstract_mdp.ground_actions_of('move(x3,x1)'))
        self.assertEqual({'move(b1,table)'}, abstract_mdp.ground_actions_of('move(x1,table)'))

        # This abstract action should randomly result in one of 2 next ground states
        next_state, next_reward = abstract_mdp.transition('move(x1,x3)')
        next_ground_state_option_1 = {'on(b1,b3)', 'on(b2,table)', 'on(b3,table)', 'on(b4,table)'}
        next_ground_state_option_2 = {'on(b1,b4)', 'on(b2,table)', 'on(b3,table)', 'on(b4,table)'}

        opt1 = (mdp.state == next_ground_state_option_1)
        opt2 = (mdp.state == next_ground_state_option_2)
        self.assertTrue((opt1 and (not opt2)) or ((not opt1) and opt2))


    def test_background_knowledge(self):

        background_knowledge = """
            nrTowers(N) :- N = #count { block(B): on(B,table) }.
        """

        rules = [

            """
            :- nrTowers(N), N != 2.
            """,
        ]

        # A state with a tower count of 1 should end up in the gutter
        mdp = BlocksWorld(state_initial={'on(b1,b2)', 'on(b2,table)'}, 
                          state_static={'subgoal(b2,b1)'})
        abstract_mdp = Carcass(mdp, rules, background_knowledge)
        self.assertEqual('gutter', abstract_mdp.state)

        # A state with a tower count of 2 should end up in abstract state 0
        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,table)'}, 
                          state_static={'subgoal(b2,b1)'})
        abstract_mdp = Carcass(mdp, rules, background_knowledge)
        self.assertEqual('abstract0', abstract_mdp.state)
