import os
import sys
import unittest

# Make sure the path of the framework is included in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Framework imports
from mdp import BlocksWorld, BlocksWorldBuilder, VacuumCleanerWorld, VacuumCleanerWorldBuilder, Sokoban, SokobanBuilder
from policy import PlannerPolicy

class TestPlanner(unittest.TestCase): 

    def test_blocksworld_1(self): 

        mdp_builder = BlocksWorldBuilder(blocks_world_size=2)
        mdp = BlocksWorld(state_initial={'on(b0,table)', 'on(b1,table)'},
                          state_static={'subgoal(b1,b0)'})

        planner = PlannerPolicy(planning_horizon=1, mdp_builder=mdp_builder)

        suggested_action, expected_return = planner.suggest_action_and_return_for_ground_state(mdp.state)

        self.assertEqual(suggested_action, planner.suggest_action_for_ground_state(mdp.state))

        mdp.transition(suggested_action)

        self.assertEqual('move(b1,b0)', suggested_action)
        self.assertEqual(mdp.return_history[0], expected_return)

    def test_blocksworld_2(self):
        
        mdp_builder = BlocksWorldBuilder(blocks_world_size=2)
        mdp = BlocksWorld(state_initial={'on(b0,b1)', 'on(b1,table)'},
                          state_static={'subgoal(b1,b0)'})

        planner = PlannerPolicy(planning_horizon=2, mdp_builder=mdp_builder)


        suggested_action_0, expected_return_0 = planner.suggest_action_and_return_for_ground_state(mdp.state)
        mdp.transition(suggested_action_0)

        suggested_action_1, expected_return_1 = planner.suggest_action_and_return_for_ground_state(mdp.state)
        mdp.transition(suggested_action_1)

        self.assertEqual('move(b0,table)', suggested_action_0)
        self.assertEqual(mdp.return_history[0], expected_return_0)

        self.assertEqual('move(b1,b0)', suggested_action_1)
        self.assertEqual(mdp.return_history[1], expected_return_1)

    def test_blocksworld_optimal_return(self):

        mdp_builder = BlocksWorldBuilder(blocks_world_size=5)
        mdp = BlocksWorld(state_initial={'on(b2,b1)', 'on(b0,b3)', 'on(b4,table)', 'on(b1,table)', 
                                         'on(b3,table)'},
                          state_static={'subgoal(b0,table)', 'subgoal(b1,b0)', 'subgoal(b2,b1)', 
                                        'subgoal(b3,b2)', 'subgoal(b4,b3)'})

        planner = PlannerPolicy(planning_horizon=2*5+1, mdp_builder=mdp_builder)

        self.assertEqual(94, planner.compute_optimal_return_for_ground_state(mdp.state))


    def test_vacuum_cleaner_world(self):

        builder = VacuumCleanerWorldBuilder()
        mdp = VacuumCleanerWorld()

        planner = PlannerPolicy(planning_horizon=4, mdp_builder=builder)

        suggested_action_0, expected_return_0 = planner.suggest_action_and_return_for_ground_state(mdp.state)
        self.assertEqual(suggested_action_0, planner.suggest_action_for_ground_state(mdp.state))
        mdp.transition(suggested_action_0)

        suggested_action_1, expected_return_1 = planner.suggest_action_and_return_for_ground_state(mdp.state)
        self.assertEqual(suggested_action_1, planner.suggest_action_for_ground_state(mdp.state))
        mdp.transition(suggested_action_1)

        suggested_action_2, expected_return_2 = planner.suggest_action_and_return_for_ground_state(mdp.state)
        self.assertEqual(suggested_action_2, planner.suggest_action_for_ground_state(mdp.state))
        mdp.transition(suggested_action_2)

        self.assertEqual('vacuum', suggested_action_0)
        self.assertEqual(mdp.return_history[0], expected_return_0)

        self.assertEqual('move(right)', suggested_action_1)
        self.assertEqual(mdp.return_history[1], expected_return_1)

        self.assertEqual('vacuum', suggested_action_2)
        self.assertEqual(mdp.return_history[2], expected_return_2)

    def test_vacuum_cleaner_world_optimal_return(self):

        builder = VacuumCleanerWorldBuilder()
        mdp = VacuumCleanerWorld()
        planner = PlannerPolicy(planning_horizon=4, mdp_builder=builder)

        self.assertEqual(-1 + -1 + 99, planner.compute_optimal_return_for_ground_state(mdp.state))

    def test_sokoban_1(self):

        builder = SokobanBuilder('suitcase-05-01')
        mdp = builder.build_mdp()

        planner = PlannerPolicy(planning_horizon=7, mdp_builder=builder)

        s0 = mdp.state

        a0, g0 = planner.suggest_action_and_return_for_ground_state(mdp.state)
        self.assertEqual(g0, 94)

        mdp.transition('push(6,3,left)')
        s1 = s0 - { 'sokoban(4,3)', 'box(6,3)' } | { 'sokoban(6,3)', 'box(5,3)' }
        self.assertSetEqual(s1, mdp.state)
        
        a1, g1 = planner.suggest_action_and_return_for_ground_state(mdp.state)
        self.assertEqual(g1, 95)

    def test_sokoban_2(self):
        builder = SokobanBuilder('suitcase-05-01')
        mdp = builder.build_mdp()

        planner = PlannerPolicy(planning_horizon=6, mdp_builder=builder)

        suggested_actions = []
        suggested_returns = []

        for i in range(20): # 20 is intentionally set to be higher than the number of needed moves.

            if len(mdp.available_actions) > 0:

                a, g = planner.suggest_action_and_return_for_ground_state(mdp.state)

                self.assertTrue(a in mdp.available_actions)

                mdp.transition(a)
                suggested_actions += [a]
                suggested_returns += [g]

        self.assertEqual(suggested_actions, mdp.action_history)
        self.assertEqual(suggested_returns + [0], mdp.return_history)

    def test_sokoban_3(self):

        builder = SokobanBuilder(level_name='suitcase-05-02')
        mdp = builder.build_mdp()

        planner = PlannerPolicy(planning_horizon=6, mdp_builder=builder)


        suggested_actions = []
        suggested_returns = []

        for i in range(20): # 20 is intentionally set to be higher than the number of needed moves.

            if len(mdp.available_actions) > 0:

                a, g = planner.suggest_action_and_return_for_ground_state(mdp.state)

                self.assertNotEqual(None, a)
                self.assertNotEqual(set(), mdp.available_actions)
                self.assertTrue(a in mdp.available_actions)

                mdp.transition(a)
                suggested_actions += [a]
                suggested_returns += [g]

        self.assertEqual(suggested_actions, mdp.action_history)
        self.assertEqual(suggested_returns + [0], mdp.return_history)

    def test_sokoban_4(self):

        # Sometimes, it is possible to end up in nonterminal states where 
        # reaching the goal is no longer possible.
        # In these cases, the planner should still yield the least harmful
        # next action!

        builder = SokobanBuilder(level_name='suitcase-05-01d')
        mdp = builder.build_mdp()

        self.assertSetEqual({'push(3,2,left)'}, mdp.available_actions)

        planner = PlannerPolicy(planning_horizon=3, mdp_builder=builder)
        a, g = planner.suggest_action_and_return_for_ground_state(mdp.state)

        self.assertEqual('push(3,2,left)', a)
        self.assertEqual(-101, g)

    def test_sokoban_5(self):

        # Sometimes, it is possible to end up in nonterminal states where 
        # reaching the goal is no longer possible.
        # In these cases, the planner should still yield the least harmful
        # next action!

        builder = SokobanBuilder(level_name='suitcase-05-04a')
        mdp = builder.build_mdp()

        self.assertSetEqual({'push(6,4,right)'}, mdp.available_actions)

        planner = PlannerPolicy(planning_horizon=2, mdp_builder=builder)
        a, g = planner.suggest_action_and_return_for_ground_state(mdp.state)

        self.assertEqual('push(6,4,right)', a)
        self.assertEqual(-101, g)
