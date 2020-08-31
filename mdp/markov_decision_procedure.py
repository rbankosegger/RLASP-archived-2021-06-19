import os
import clingo
import random

from typing import Set, List

class MarkovDecisionProcedure:

    @staticmethod
    def file_path(file_name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

    def __init__(self, initial_state: Set[str], goal_state: Set[str], discount_rate: float,
                 asp_file_name: str):

        self.state: Set[str] = frozenset(initial_state)
        self.goal_state: Set[str] = frozenset(goal_state)
        self.discount_rate: float = discount_rate

        # TODO: Needs to be separated from abstract MDP. -> Do it when introducing a second MDP
        self.interface_file_name :str = 'markov_decision_procedure.lp'
        self.file_name: str = asp_file_name

        # MDP trajectory: S0, A0, R1, S1, A1, R2, S2, A2, ... 
        self.state_history: List[Set[str]] = [frozenset(initial_state)] # S0
        self.action_history: List[str] = [] #A0 will be given later once the first action is executed
        self.reward_history: List[float] = [None] # R0, which is undefined

        self.available_actions = self._compute_available_actions()

    def transition(self, action: str):
        
        ctl = clingo.Control()

        ctl.load(self.file_path(self.interface_file_name))
        ctl.load(self.file_path(self.file_name))
        ctl.add('base', [], ' '.join(f'current({s}).' for s in self.state))
        ctl.add('base', [], ' '.join(f'subgoal({s}).' for s in self.goal_state))
        ctl.add('base', [], '#const t=1.')
        ctl.add('base', [], f'action({action}).')
        ctl.add('base', [], '#show state/1. #show nextReward/1. #show executable/1.')

        ctl.ground(parts=[('base', [])])
        models = ctl.solve(yield_=True)

        # Since we are only modelling deterministic actions, there is only one possible next state (model).
        model = next(models)

        next_reward = None
        next_state = set()
        available_actions = set()

        for symbol in model.symbols(shown=True):

            if symbol.name == 'state':

                #˙Atom is of the form `state(f(...))` 
                # where`f(...)` is an uninterpreted function belonging to the state representation.
                f = symbol.arguments[0]
                next_state.add(str(f))

            if symbol.name == 'nextReward':

                # Atom is of the form `nextReward(r)`, and `r` is the reward.
                next_reward = symbol.arguments[0].number

            if symbol.name == 'executable':

                # Atom is of the form `executable(f(...))` 
                # where`f(...)` is an uninterpreted function representing an executable action.
                available_actions.add(str(symbol.arguments[0]))

        self.state = frozenset(next_state)
        self.available_actions = available_actions

        # Update trajectory:
        self.action_history.append(action) # A[t]
        self.state_history.append(frozenset(next_state)) # S[t+1]
        self.reward_history.append(next_reward) # R[t+1]


    def compute_optimal_return(self, max_planning_horizon: int = None) -> float:

        if max_planning_horizon is None:
            max_planning_horizon = 2*len(self.state)

        ctl = clingo.Control()

        ctl.load(self.file_path(self.interface_file_name))
        ctl.load(self.file_path(self.file_name))
        ctl.add('base', [], ' '.join(f'current({s}).' for s in self.state))
        ctl.add('base', [], ' '.join(f'subgoal({s}).' for s in self.goal_state))
        ctl.add('base', [], f'#const t={max_planning_horizon}.')
        ctl.add('base', [], '#show maxReturn/1.')

        ctl.configuration.solve.models = 0  # create all stable models and find the optimal one
        ctl.ground(parts=[('base', [])])
        models = ctl.solve(yield_=True)

        model = list(models)[0] # [maxReturn(r)]
        symbol = model.symbols(shown=True)[0] # maxReturn(r)
        optimal_return = float(symbol.arguments[0].number) # r

        return optimal_return 


    @property
    def return_history(self) -> List[float]:

        T = len(self.state_history)
        G = [0] * T

        for t in reversed(range(T-1)):
            G[t] = self.reward_history[t+1] + self.discount_rate * G[t+1]

        return G

    def _compute_available_actions(self) -> Set[str]:

        ctl = clingo.Control()

        ctl.load(self.file_path(self.interface_file_name))
        ctl.load(self.file_path(self.file_name))
        ctl.add('base', [], ' '.join(f'current({s}).' for s in self.state))
        ctl.add('base', [], ' '.join(f'subgoal({s}).' for s in self.goal_state))
        ctl.add('base', [], '#const t=0.')
        ctl.add('base', [], '#show executable/1.')

        ctl.ground(parts=[('base', [])])
        models = ctl.solve(yield_=True)

        # In search for next actions, we only expect one answer set.
        model = next(models)

        available_actions = set()
        for symbol in model.symbols(shown=True):

            # We expect atoms of the form `executable(move(X, Y)` 
            # but we are only interested in the first argument `move(X, Y)`
            available_actions.add(str(symbol.arguments[0]))

        return available_actions

