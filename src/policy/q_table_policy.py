from typing import Set, Dict, Any
import random

from . import RandomPolicy
from mdp import MarkovDecisionProcedure


class QTablePolicy:

    def __init__(self, initial_value_estimate: float = 0.0):
        self._q_table: Dict[Any, Dict[Any, float]] = dict()
        self.initial_value_estimate: float = initial_value_estimate

    def is_new_state(self, state) -> bool:
        return not state in self._q_table 

    def value_for(self, state, action) -> float:

        if action is None:
            return 0

        return self._q_table[state][action]

    def suggest_action_for_state(self, state, *args) -> Any:

        available_estimates = self._q_table[state].items()

        if len(available_estimates) == 0:
            return None

        current_maximal_estimate = max(v for _,v in available_estimates) 

        current_optimal_actions = [a for (a, v) in available_estimates 
                                   if v==current_maximal_estimate]

        return random.choice(current_optimal_actions)

    def initialize_state(self, state, available_actions: Set):
        self._q_table[state] = { a: self.initial_value_estimate for a in available_actions } 

    def update(self, state, action, delta:float):
        self._q_table[state][action] += delta

    def optimal_value_for(self, state):
        return self.value_for(state, self.suggest_action_for_state(state))
