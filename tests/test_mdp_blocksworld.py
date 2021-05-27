import os
import sys
import unittest

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

# Framework imports
from mdp import BlocksWorld, BlocksWorldBuilder

class TestBlocksWorld(unittest.TestCase):


    def test_available_actions_1(self):

        mdp = BlocksWorld(state_initial={'on(b2,table)', 'on(b1,b2)'},
                          state_static={'subgoal(b2,b1)'})
        self.assertSetEqual({'move(b1,table)'}, 
                         mdp.available_actions)

    def test_available_actions_2(self):

        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,table)'},
                          state_static={'subgoal(b2,b1)'})
        self.assertSetEqual({'move(b1,b2)', 'move(b2,b1)'}, 
                         mdp.available_actions)

    def test_available_actions_3(self):

        mdp = BlocksWorld(state_initial={'on(b2,table)', 'on(b1,b2)', 'on(b3,table)'},
                          state_static={'subgoal(b2,b1)'})
        self.assertEqual({'move(b1,table)', 'move(b1,b3)', 'move(b3,b1)'}, 
                         mdp.available_actions)

    def test_available_actions_4(self):

        mdp = BlocksWorld(state_initial={'on(b2,table)', 'on(b1,table)', 'on(b3,table)'},
                          state_static={'subgoal(b2,b1)'})

        # Available actions should be updated after state transitions.
        mdp.transition('move(b1,b2)')
        self.assertEqual({'move(b1,table)', 'move(b1,b3)', 'move(b3,b1)'}, 
                         mdp.available_actions)

    def test_available_actions_5(self):

        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,b1)'},
                          state_static={'subgoal(b2,b1)'})

        # No actions available in the goal state
        self.assertEqual(set(), 
                         mdp.available_actions)

    def test_state_transition(self):

        # Transition to arbitrary state
        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,b1)'},
                          state_static={'subgoal(b1,b2)'})

        next_state, next_reward = mdp.transition('move(b2,table)')
        self.assertEqual({'on(b1,table)', 'on(b2,table)'}, next_state)
        self.assertEqual(-1, next_reward)
        self.assertEqual({'on(b1,table)', 'on(b2,table)'}, mdp.state)

        next_state, next_reward = mdp.transition('move(b1,b2)')
        self.assertEqual({'on(b1,b2)', 'on(b2,table)', 'goal'}, next_state)
        self.assertEqual(99, next_reward)
        self.assertEqual({'on(b1,b2)', 'on(b2,table)', 'goal'}, mdp.state)

        # Check if trajectory is correct: S0, A0, R1, S1, A1, R2, S2
        self.assertEqual({'on(b1,table)', 'on(b2,b1)'}, mdp.state_history[0]) # S0
        self.assertEqual('move(b2,table)', mdp.action_history[0]) # A0
        self.assertEqual(-1, mdp.reward_history[1]) # R1
        self.assertEqual({'on(b1,table)', 'on(b2,table)'}, mdp.state_history[1]) #S1
        self.assertEqual('move(b1,b2)', mdp.action_history[1]) # A1
        self.assertEqual(100-1, mdp.reward_history[2]) # R2
        self.assertEqual({'on(b1,b2)', 'on(b2,table)', 'goal'}, mdp.state_history[2]) #S2

    def test_returns_1(self):

        # Optimal way to goal
        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,table)'},
                          state_static = {'subgoal(b2,b1)'})
        mdp.transition('move(b2,b1)')

        # G[t] = R[t+1] + ...
        self.assertEqual(mdp.return_history[0], 99)


    def test_returns_2(self):

        # Undiscounted return

        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,table)'},
                          state_static = {'subgoal(b2,b1)'})
        mdp.transition('move(b1,b2)')
        mdp.transition('move(b1,table)')
        mdp.transition('move(b2,b1)')

        # G[t] = R[t+1] + R[t+2] + R[t+3]
        self.assertEqual(mdp.return_history[0], -1 + -1 + 99)
        self.assertEqual(mdp.return_history[1], -1 + 99)
        self.assertEqual(mdp.return_history[2], 99)
        self.assertEqual(mdp.return_history[3], 0) # Return is zero in terminal state

    def test_ground_state(self):

        mdp = BlocksWorld(state_initial={'on(b1,table)', 'on(b2,table)'},
                          state_static = {'subgoal(b2,b1)'})

        self.assertEqual({'on(b1,table)', 'on(b2,table)'}, mdp.ground_state)
