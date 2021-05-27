import os
import sys
import unittest

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

# Framework imports
from mdp import VacuumCleanerWorld 

class TestVacuumCleanerWorld(unittest.TestCase):

    def test_available_actions_1(self):

        mdp = VacuumCleanerWorld() 
        self.assertSetEqual({'move(right)', 'vacuum'}, mdp.available_actions)

    def test_state_transition_1(self):

        mdp = VacuumCleanerWorld()

        next_state, next_reward = mdp.transition('move(right)')

        self.assertSetEqual({'robot(right)','dirty(left)', 'dirty(right)'}, mdp.state)
        self.assertSetEqual({'move(left)', 'vacuum'}, mdp.available_actions)
        self.assertSetEqual({'robot(right)','dirty(left)', 'dirty(right)'}, next_state)
        self.assertEqual(-1, next_reward)

    def test_state_transition_2(self):

        mdp = VacuumCleanerWorld()

        mdp.transition('vacuum')

        self.assertSetEqual({'robot(left)', 'dirty(right)'}, mdp.state)
        self.assertSetEqual({'move(right)'}, mdp.available_actions)

    def test_returns(self):

        mdp = VacuumCleanerWorld()
        mdp.transition('move(right)')
        mdp.transition('vacuum')
        mdp.transition('move(left)')
        mdp.transition('vacuum')

        self.assertEqual(mdp.return_history[0], -1 + -1 + -1 + 99)
        self.assertEqual(mdp.return_history[1], -1 + -1 + 99)
        self.assertEqual(mdp.return_history[2], -1 + 99)
        self.assertEqual(mdp.return_history[3], 99)

    def test_trajectory(self):

        mdp = VacuumCleanerWorld()
        mdp.transition('vacuum')
        mdp.transition('move(right)')
        mdp.transition('vacuum')

        # Check if trajectory is correct: S0, A0, R1, S1, A1, R2, S2, A2, R3, S3
        self.assertEqual({'robot(left)', 'dirty(left)', 'dirty(right)'}, mdp.state_history[0]) # S0
        self.assertEqual('vacuum', mdp.action_history[0]) # A0
        self.assertEqual(-1, mdp.reward_history[1]) # R1
        self.assertEqual({'robot(left)', 'dirty(right)'}, mdp.state_history[1]) #S1
        self.assertEqual('move(right)', mdp.action_history[1]) # A1
        self.assertEqual(-1, mdp.reward_history[2]) # R2
        self.assertEqual({'robot(right)', 'dirty(right)'}, mdp.state_history[2]) #S2
        self.assertEqual(99, mdp.reward_history[3]) # R3
        self.assertEqual({'robot(right)'}, mdp.state_history[3]) #S3

    def test_ground_state(self):

        mdp = VacuumCleanerWorld()
        self.assertEqual({'robot(left)', 'dirty(left)', 'dirty(right)'}, mdp.ground_state)
