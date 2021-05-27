import random

from . import RandomPolicy, PlannerPolicy, QTablePolicy

class PlanningExploringStartsPolicy:

    def __init__(self, planner_policy: PlannerPolicy, random_policy: RandomPolicy, 
                 qtable_policy: QTablePolicy, 
                 planning_factor: float = 0.0,
                 plan_for_new_states: float = False):

        self.planner_policy = planner_policy
        self.random_policy = random_policy 
        self.qtable_policy = qtable_policy

        self.planning_factor = planning_factor
        self.plan_for_new_states = plan_for_new_states

        self.first_action_in_episode = True

        self.known_states = set()

    def is_new_state(self, state):
        return any(p.is_new_state(state) for p in [self.random_policy, self.qtable_policy])

    def initialize_state(self, state, available_actions):
        self.random_policy.initialize_state(state, available_actions)
        self.qtable_policy.initialize_state(state, available_actions)

    def initialize_new_episode(self):
        self.first_action_in_episode = True

    def suggest_action_for_state(self, state, ground_state):

        is_new_episode = self.first_action_in_episode
        is_new_state = not state in self.known_states

        self.first_action_in_episode &= False
        self.known_states.add(state)

        if is_new_episode:

            # The very first action is an exploring start.
            return self.random_policy.suggest_action_for_state(state)

        elif is_new_state:

            if self.plan_for_new_states:
                return self.planner_policy.suggest_action_for_ground_state(ground_state)

            else:
                return self.qtable_policy.suggest_action_for_state(state)
            
        else:

            if random.random() >= self.planning_factor:
                return self.qtable_policy.suggest_action_for_state(state)

            else:
                return self.planner_policy.suggest_action_for_ground_state(ground_state)

    def update(self, state, action, delta: float):
        self.qtable_policy.update(state, action, delta)

    def value_for(self, state, action):
        return self.qtable_policy.value_for(state, action)

    def optimal_value_for(self, state):
        return self.qtable_policy.optimal_value_for(state)
