import os
import clingo
from typing import Tuple, Set

from mdp import MarkovDecisionProcedure

class PlannerPolicy:

    def __init__(self, planning_horizon: int, mdp_builder):

        self.planning_horizon: int = planning_horizon
        self.mdp_interface_file_path: str = mdp_builder.mdp_interface_file_path
        self.mdp_problem_file_path: str = mdp_builder.mdp_problem_file_path
        self.mdp_state_static: str = mdp_builder.mdp_state_static

        self.planner_file_name: str = 'planner_policy.dl'
        self.planner_file_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.planner_file_name)

        self.asp_output = None


    def suggest_action_for_ground_state(self, ground_state) -> str:
        next_action, _ = self.suggest_action_and_return_for_ground_state(ground_state)
        return next_action

    def compute_optimal_return_for_ground_state(self, ground_state):
        _, optimal_return = self.suggest_action_and_return_for_ground_state(ground_state)
        return optimal_return

    def suggest_action_and_return_for_ground_state(self, ground_state) -> Tuple[str, int]:

        ctl = clingo.Control()

        ctl.load(self.mdp_interface_file_path)
        ctl.load(self.mdp_problem_file_path)
        ctl.load(self.planner_file_path)
        ctl.add('base', [], ' '.join(f'currentState({s}).' for s in ground_state))
        ctl.add('base', [], ' '.join(f'{s}.' for s in self.mdp_state_static))
        ctl.add('base', [], f'#const t={self.planning_horizon}.')
        ctl.add('base', [], '#show maxReturn/1. #show bestCurrentAction/1.')

        ctl.configuration.solve.models = 0  # create all stable models and find the optimal one
        ctl.ground(parts=[('base', [])])
        models = ctl.solve(yield_=True)

        model = list(models)[0]
        self.asp_output = str(model)

        expected_return = None
        suggested_action = None
        
        for symbol in model.symbols(shown=True):
            if symbol.name == 'maxReturn':

                # Atom is of the form `maxReward(r)` and `r` is the expected return of the current state.

                expected_return = symbol.arguments[0].number 

            if symbol.name == 'bestCurrentAction':

                # Atom is of the form `bestCurrentAction(f(...))` and `f(...)` is an uninterpreted function,
                # corresponding to the suggested action

                suggested_action = str(symbol.arguments[0])

        return (suggested_action, expected_return)

    def initialize_new_episode(self):
        # Nothing to prepare in this policy
        pass
