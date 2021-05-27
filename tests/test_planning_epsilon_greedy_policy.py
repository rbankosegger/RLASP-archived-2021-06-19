import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Framework imports
from mdp import VacuumCleanerWorldBuilder
from policy import PlanningEpsilonGreedyPolicy, PlannerPolicy, RandomPolicy, RandomPolicy, QTablePolicy
import random

class TestPlanningEpsilonGreedyPolicy(unittest.TestCase):

    def test_is_new_state(self):

        qtable_policy = QTablePolicy()
        random_policy = RandomPolicy()

        mdp_builder = VacuumCleanerWorldBuilder()
        mdp = mdp_builder.build_mdp()
        planner_policy = PlannerPolicy(1, mdp_builder)
        

        policy = PlanningEpsilonGreedyPolicy(planner_policy, random_policy, qtable_policy,
                                             epsilon=0.5)

        self.assertTrue(policy.is_new_state(state='s1'))
        policy.initialize_state(state='s1', available_actions={'a(1)', 'a(2)'})
        self.assertFalse(policy.is_new_state('s1'))

    def test_planning_in_new_states(self):

        planner_policy = MagicMock()
        planner_policy.suggest_action_for_ground_state = MagicMock(return_value='a2')

        qtable_policy = MagicMock(spec=QTablePolicy())
        random_policy = MagicMock(spec=RandomPolicy())
        policy = PlanningEpsilonGreedyPolicy(planner_policy, random_policy, qtable_policy,
                                             epsilon=0.5)

        # If a state is encountered for the first time, the planner should be called.
        policy.initialize_state(state='s1', available_actions={'a1', 'a2', 'a3'})
        suggested_action = policy.suggest_action_for_state(state='s1', ground_state='gs1')
        self.assertEqual('a2', suggested_action)
        planner_policy.suggest_action_for_ground_state.assert_called_with('gs1')

        # If a state is encountered more than once, the other policies are used.
        planner_policy.suggest_action_for_state.reset_mock()
        suggested_action = policy.suggest_action_for_state(state='s1', ground_state='gs1')
        self.assertTrue(qtable_policy.suggest_action_for_state.called \
                        or random_policy.suggest_action_for_state.called)
        planner_policy.suggest_action_for_state.assert_not_called()


    def test_no_planning_in_new_states(self):

        planner_policy = MagicMock()
        planner_policy.suggest_action_for_state = MagicMock(return_value='plan')

        qtable_policy = MagicMock()
        qtable_policy.suggest_action_for_state = MagicMock(return_value='qtable')

        random_policy = MagicMock()
        random_policy.suggest_action_for_state = MagicMock(return_value='random')

        policy = PlanningEpsilonGreedyPolicy(planner_policy, random_policy, qtable_policy, epsilon=0.5, 
                                              plan_for_new_states=False)

        # If a state is encountered for the first time, other policies should be used.
        policy.initialize_state(state='s1', available_actions={'plan', 'qtable', 'random'})
        suggested_action = policy.suggest_action_for_state(state='s1', ground_state='gs1')
        self.assertIn(suggested_action, {'plan', 'qtable', 'random'})
        self.assertTrue(qtable_policy.suggest_action_for_state.called \
                        or random_policy.suggest_action_for_state.called)
        planner_policy.suggest_action_for_state.assert_not_called()



    def test_epsilon(self):

        planner_policy = MagicMock()
        qtable_policy = MagicMock(spec=QTablePolicy())
        random_policy = MagicMock(spec=RandomPolicy())
        policy = PlanningEpsilonGreedyPolicy(planner_policy, random_policy, qtable_policy,
                                             epsilon=0.3)

        policy.initialize_state(state='s1', available_actions={'a1', 'a2', 'a3'})

        # The first suggestion comes from the planner
        policy.suggest_action_for_state(state='s1', ground_state='gs1')
        planner_policy.reset_mock()

        # The second suggestion is randomly chosen between the qtable policy and 
        # the random policy. Let's mock up the random number generator to only 
        # get suggestions from the qtable policy.
        with patch('random.random', MagicMock(return_value=0.31)): # 0.31 > 0.3 -> follow greedy policy
            policy.suggest_action_for_state(state='s1', ground_state='gs1')

            qtable_policy.suggest_action_for_state.assert_called_with('s1')
            planner_policy.suggest_action_for_state.assert_not_called()
            random_policy.suggest_action_for_state.assert_not_called()

        # On the other hand, when we mock up the random number generator such
        # that we only get suggestions from the random policy, things should
        # go accordingly.
        qtable_policy.reset_mock()
        planner_policy.reset_mock()
        random_policy.reset_mock()
        with patch('random.random', MagicMock(return_value=0.29)): #0.29 < 0.3 -> follow random policy
            policy.suggest_action_for_state(state='s1', ground_state='gs1')

            random_policy.suggest_action_for_state.assert_called_with('s1')
            planner_policy.suggest_action_for_state.assert_not_called()
            qtable_policy.suggest_action_for_state.assert_not_called()


    def test_update(self):

        planner_policy = MagicMock()
        qtable_policy = MagicMock(spec=QTablePolicy())
        random_policy = MagicMock(spec=RandomPolicy())
        policy = PlanningEpsilonGreedyPolicy(planner_policy, random_policy, qtable_policy,
                                             epsilon=0.3)

        # Updating the policy should update the qtable policy as well.
        policy.initialize_state('s1', {'a1', 'a2'})
        policy.update('s1', 'a2', -1.23)
        qtable_policy.update.assert_called_with('s1', 'a2', -1.23)

    def test_value_for(self):
        planner_policy = MagicMock()
        qtable_policy = QTablePolicy()
        random_policy = RandomPolicy()
        policy = PlanningEpsilonGreedyPolicy(planner_policy, random_policy, qtable_policy,
                                             epsilon=0.3)

        # Evaluation of a state-action pair should be the same as for the qtable policy.
        policy.initialize_state('s1', {'a1', 'a2'})
        policy.update('s1', 'a2', -1.23)
        self.assertEqual(-1.23, policy.value_for('s1', 'a2'))

    def test_optimal_value_for(self):
        
        planner_policy = MagicMock()
        qtable_policy = QTablePolicy()
        random_policy = RandomPolicy()
        policy = PlanningEpsilonGreedyPolicy(planner_policy, random_policy, qtable_policy,
                                             epsilon=0.3)

        # Evaluation of a state-action pair should be the same as for the qtable policy.
        policy.initialize_state('s', {'a', 'b', 'c'})
        policy.update('s', 'a', -1.23)
        policy.update('s', 'b', 5.43)
        policy.update('s', 'c', 0.03)
        self.assertEqual(5.43, policy.optimal_value_for('s'))
