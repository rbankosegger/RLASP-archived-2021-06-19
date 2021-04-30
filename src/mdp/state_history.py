from typing import Set, List

class StateHistory:

    def __init__(self, state_initial):

        # MDP trajectory: S0, A0, R1, S1, A1, R2, S2, A2, ... 
        self.state_history: List[Set[str]] = [state_initial] # S0
        self.action_history: List[str] = [] #A0 will be given later once the first action is executed
        self.reward_history: List[float] = [None] # R0, which is undefined

    def transition(self, action, next_state, next_reward):

        # Update trajectory:
        self.action_history.append(action) # A[t]
        self.state_history.append(next_state) # S[t+1]
        self.reward_history.append(next_reward) # R[t+1]


    @property
    def return_history(self) -> List[float]:

        T = len(self.state_history)
        G = [0] * T

        for t in reversed(range(T-1)):
            G[t] = self.reward_history[t+1] + self.discount_rate * G[t+1]

        return G
