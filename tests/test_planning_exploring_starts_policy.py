import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Framework imports
from mdp import VacuumCleanerWorldBuilder
from policy import PlanningExploringStartsPolicy, PlannerPolicy, RandomPolicy, RandomPolicy, QTablePolicy
import random

class TestPlanningExploringStartsPolicy(unittest.TestCase):

    def test_is_new_state(self):

        qtable_policy = QTablePolicy()
        random_policy = RandomPolicy()

        mdp_builder = VacuumCleanerWorldBuilder()
        mdp = mdp_builder.build_mdp()
        planner_policy = PlannerPolicy(planning_horizon=1, mdp_builder=mdp_builder)
        

        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy)

        self.assertTrue(policy.is_new_state(state='s1'))
        policy.initialize_state(state='s1', available_actions={'a(1)', 'a(2)'})
        self.assertFalse(policy.is_new_state('s1'))

    def test_exploring_starts(self):

        planner_policy = MagicMock(spec=PlannerPolicy)
        qtable_policy = MagicMock(spec=QTablePolicy)
        random_policy = MagicMock(spec=RandomPolicy)
        random_policy.suggest_action_for_state = MagicMock(return_value='a3')

        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy)
        policy.initialize_state('s1', {'a1', 'a2', 'a3'})

        suggested_action = policy.suggest_action_for_state(state='s1', ground_state='gs1')

        # The first action in a new episode should be random (exploring start) 
        random_policy.suggest_action_for_state.assert_called_with('s1')
        self.assertEqual('a3', suggested_action)

        # Any following action should either be planned or greedy, but not random.
        random_policy.reset_mock()
        policy.initialize_state('s2', {'a3', 'a4', 'a5'})
        _ = policy.suggest_action_for_state(state='s2', ground_state='gs2')
        self.assertTrue(planner_policy.suggest_action_for_ground_state.called \
                        or qtable_policy.suggest_action_for_state.called)
        random_policy.suggest_action_for_state.assert_not_called()

        # If we start a new episode, the first action should be random again.
        policy.initialize_new_episode()
        suggested_action = policy.suggest_action_for_state(state='s1', ground_state='gs1')
        random_policy.suggest_action_for_state.assert_called_with('s1')
        self.assertEqual('a3', suggested_action)
        
    def test_plan_for_new_states(self):

        planner_policy = MagicMock(spec=PlannerPolicy)
        planner_policy.suggest_action_for_ground_state = MagicMock(return_value='a2')
        qtable_policy = MagicMock(spec=QTablePolicy)
        qtable_policy.suggest_action_for_state = MagicMock(return_value='a3')
        random_policy = MagicMock(spec=RandomPolicy)

        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy,
                                               plan_for_new_states = True)
        
        # First action is an exploring start
        policy.initialize_state('s1', {'a1'})
        policy.suggest_action_for_state(state='s1', ground_state='gs1')
        random_policy.reset_mock()

        # Second action is in an unknown state. The planner should be used.
        policy.initialize_state('s2', {'a2', 'a3'})
        suggested_action = policy.suggest_action_for_state(state='s2', ground_state='gs2')
        self.assertEqual('a2', suggested_action)
        planner_policy.suggest_action_for_ground_state.assert_called_with('gs2')
        random_policy.suggest_action_for_state.assert_not_called()
        qtable_policy.suggest_action_for_state.assert_not_called()
        planner_policy.reset_mock()

        # The next time we encounter s2, no planning should happen.
        suggested_action = policy.suggest_action_for_state(state='s2', ground_state='gs2')
        self.assertEqual('a3', suggested_action)
        qtable_policy.suggest_action_for_state.assert_called_with('s2')
        random_policy.suggest_action_for_state.assert_not_called()
        planner_policy.suggest_action_for_ground_state.assert_not_called()

    def test_dont_plan_for_new_states(self):

        planner_policy = MagicMock(spec=PlannerPolicy)
        planner_policy.suggest_action_for_state = MagicMock(return_value='a2')
        qtable_policy = MagicMock(spec=QTablePolicy)
        qtable_policy.suggest_action_for_state = MagicMock(return_value='a3')
        random_policy = MagicMock(spec=RandomPolicy)

        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy,
                                               plan_for_new_states = False)
        
        # First action is an exploring start
        policy.initialize_state('s1', {'a1'})
        policy.suggest_action_for_state(state='s1', ground_state='gs1')
        random_policy.reset_mock()

        # Second action is in an unknown state. The planner should be used.
        policy.initialize_state('s2', {'a2', 'a3'})
        suggested_action = policy.suggest_action_for_state(state='s2', ground_state='gs2')
        self.assertEqual('a3', suggested_action)
        planner_policy.suggest_action_for_state.assert_not_called()
        random_policy.suggest_action_for_state.assert_not_called()
        qtable_policy.suggest_action_for_state.assert_called_with('s2')
        planner_policy.reset_mock()

        # The next time we encounter s2, no planning should happen.
        suggested_action = policy.suggest_action_for_state(state='s2', ground_state='gs2')
        self.assertEqual('a3', suggested_action)
        qtable_policy.suggest_action_for_state.assert_called_with('s2')
        random_policy.suggest_action_for_state.assert_not_called()
        planner_policy.suggest_action_for_state.assert_not_called()

    def test_planning_factor(self):

        planner_policy = MagicMock(spec=PlannerPolicy)
        planner_policy.suggest_action_for_ground_state = MagicMock(return_value='plan')
        qtable_policy = MagicMock(spec=QTablePolicy)
        qtable_policy.suggest_action_for_state = MagicMock(return_value='greedy')
        random_policy = MagicMock(spec=RandomPolicy)

        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy,
                                               planning_factor = 0.2)
        
        # First action is an exploring start
        policy.initialize_state('s1', {'plan', 'greedy'})
        policy.suggest_action_for_state(state='s1', ground_state='gs1')
        random_policy.reset_mock()


        # The second suggestion is randomly chosen between the qtable policy and 
        # the planning policy. Let's mock up the random number generator to only 
        # get suggestions from the qtable policy.
        with patch('random.random', MagicMock(return_value=0.21)): # 0.21 > 0.2 -> follow greedy policy
            suggested_action = policy.suggest_action_for_state(state='s1', ground_state='gs1')
            self.assertEqual('greedy', suggested_action)

            qtable_policy.suggest_action_for_state.assert_called_with('s1')
            planner_policy.suggest_action_for_ground_state.assert_not_called()
            random_policy.suggest_action_for_state.assert_not_called()
            qtable_policy.reset_mock()


        # On the other hand, when we mock up the random number generator such
        # that we only get suggestions from the planner policy, things should
        # go accordingly.
        with patch('random.random', MagicMock(return_value=0.19)): # 0.19 < 0.2 -> follow planner policy
            suggested_action = policy.suggest_action_for_state(state='s1', ground_state='gs1')
            self.assertEqual('plan', suggested_action)

            planner_policy.suggest_action_for_ground_state.assert_called_with('gs1')
            qtable_policy.suggest_action_for_state.assert_not_called()
            random_policy.suggest_action_for_state.assert_not_called()
            planner_policy.reset_mock()


    def test_update(self):

        planner_policy = MagicMock()
        qtable_policy = MagicMock(spec=QTablePolicy())
        random_policy = MagicMock(spec=RandomPolicy())
        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy)

        # Updating the policy should update the qtable policy as well.
        policy.initialize_state('s1', {'a1', 'a2'})
        policy.update('s1', 'a2', -1.23)
        qtable_policy.update.assert_called_with('s1', 'a2', -1.23)

    def test_value_for(self):
        planner_policy = MagicMock()
        qtable_policy = QTablePolicy()
        random_policy = RandomPolicy()
        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy)

        # Evaluation of a state-action pair should be the same as for the qtable policy.
        policy.initialize_state('s1', {'a1', 'a2'})
        policy.update('s1', 'a2', -1.23)
        self.assertEqual(-1.23, policy.value_for('s1', 'a2'))


    def test_optimal_value_for(self):

        planner_policy = MagicMock()
        qtable_policy = QTablePolicy()
        random_policy = RandomPolicy()
        policy = PlanningExploringStartsPolicy(planner_policy, random_policy, qtable_policy)

        # Evaluation of a state-action pair should be the same as for the qtable policy.
        policy.initialize_state('s', {'a', 'b', 'c'})
        policy.update('s', 'a', 1.23)
        policy.update('s', 'b', -5.43)
        policy.update('s', 'c', 0.03)
        self.assertEqual(1.23, policy.optimal_value_for('s'))
