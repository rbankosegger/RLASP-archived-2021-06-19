import random

from . import RandomPolicy, PlannerPolicy, QTablePolicy

class PlanningEpsilonGreedyPolicy:

    def __init__(self, planner_policy: PlannerPolicy, random_policy: RandomPolicy, 
                 qtable_policy: QTablePolicy, epsilon: float, plan_for_new_states: bool = True):

        self.random_policy = random_policy 
        self.qtable_policy = qtable_policy
        self.planner_policy = planner_policy
        self.epsilon = epsilon
        self.plan_for_new_states = plan_for_new_states

        self.planned_states = set()

    def is_new_state(self, state):
        return any(p.is_new_state(state) for p in [self.random_policy, self.qtable_policy])

    def initialize_state(self, state, available_actions):
        self.random_policy.initialize_state(state, available_actions)
        self.qtable_policy.initialize_state(state, available_actions)

    def suggest_action_for_state(self, state, ground_state):

        if not ground_state:
            ground_state = state

        if self.plan_for_new_states and not state in self.planned_states:
            self.planned_states.add(state)
            return self.planner_policy.suggest_action_for_ground_state(ground_state)
            
        else:
            if random.random() >= self.epsilon:
                return self.qtable_policy.suggest_action_for_state(state)
            else:
                return self.random_policy.suggest_action_for_state(state)

    def update(self, state, action, delta: float):
        self.qtable_policy.update(state, action, delta)

    def initialize_new_episode(self):
        # Nothing to prepare in this policy
        pass

    def value_for(self, state, action):
        return self.qtable_policy.value_for(state, action)

    def optimal_value_for(self, state):
        return self.qtable_policy.optimal_value_for(state)
