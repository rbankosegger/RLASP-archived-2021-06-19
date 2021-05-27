import os
import clingo
import random

from typing import Set, List

from .state_history import StateHistory

class MarkovDecisionProcedure(StateHistory):

    @staticmethod
    def file_path(file_name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

    def __init__(self, state_initial: Set[str], state_static: Set[str], discount_rate: float,
                 problem_file_name: str):
        
        super().__init__(frozenset(state_initial))

        self.state: Set[str] = frozenset(state_initial)
        self.state_static: Set[str] = frozenset(state_static)
        self.discount_rate: float = discount_rate

        # TODO: Needs to be separated from abstract MDP. -> Do it when introducing a second MDP
        self.interface_file_name: str = 'markov_decision_procedure.lp'
        self.problem_file_name: str = problem_file_name

        self.available_actions = self._compute_available_actions()

    @property
    def interface_file_path(self):
        return self.file_path(self.interface_file_name)

    @property
    def problem_file_path(self):
        return self.file_path(self.problem_file_name)

    @property
    def ground_state(self):
        return self.state

    def transition(self, action: str):
        
        ctl = clingo.Control()

        ctl.load(self.file_path(self.interface_file_name))
        ctl.load(self.file_path(self.problem_file_name))
        ctl.add('base', [], ' '.join(f'currentState({s}).' for s in self.state))
        ctl.add('base', [], ' '.join(f'{s}.' for s in self.state_static))
        ctl.add('base', [], f'currentAction({action}).')
        ctl.add('base', [], '#show nextState/1. #show nextReward/1. #show nextExecutable/1.')

        ctl.ground(parts=[('base', [])])
        solvehandle = ctl.solve(yield_=True)

        # Since we are only modelling deterministic actions, there is only one possible next state (model).
        model = solvehandle.model()

        next_reward = None
        next_state = set()
        available_actions = set()

        for symbol in model.symbols(shown=True):

            if symbol.name == 'nextState':

                #˙Atom is of the form `state(f(...))` 
                # where`f(...)` is an uninterpreted function belonging to the state representation.
                f = symbol.arguments[0]
                next_state.add(str(f))

            if symbol.name == 'nextReward':

                # Atom is of the form `nextReward(r)`, and `r` is the reward.
                next_reward = symbol.arguments[0].number

            if symbol.name == 'nextExecutable':

                # Atom is of the form `nextExecutable(f(...))` 
                # where`f(...)` is an uninterpreted function representing an executable action.
                available_actions.add(str(symbol.arguments[0]))

        self.state = frozenset(next_state)
        self.available_actions = available_actions

        super().transition(action, # A[t]
                           frozenset(next_state), # S[t+1]
                           next_reward # R[t+1]
                          )

        return frozenset(next_state), next_reward


    def _compute_available_actions(self) -> Set[str]:

        ctl = clingo.Control()

        ctl.load(self.file_path(self.interface_file_name))
        ctl.load(self.file_path(self.problem_file_name))
        ctl.add('base', [], ' '.join(f'currentState({s}).' for s in self.state))
        ctl.add('base', [], ' '.join(f'{s}.' for s in self.state_static))
        ctl.add('base', [], '#show currentExecutable/1.')

        ctl.ground(parts=[('base', [])])
        solvehandle = ctl.solve(yield_=True)

        # In search for next actions, we only expect one answer set.
        model = solvehandle.model()

        available_actions = set()
        for symbol in model.symbols(shown=True):

            # We expect atoms of the form `currentExecutable(move(X, Y)` 
            # but we are only interested in the first argument `move(X, Y)`
            available_actions.add(str(symbol.arguments[0]))

        return available_actions

