
class OffPolicyControl:

    def __init__(self, target_policy, behavior_policy):
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy

    def policy_update_after_step(self, current_state, current_action,
                                 next_state, next_reward, mdp):
        # This policy does not learn -> No update to policy
        return

    def policy_update_after_episode(self, mdp):
        # This policy does not learn -> No update to policy
        return

    def suggest_action_for_state(self, state, ground_state):
        # Off-Policy control uses the behavior policy for next steps.
        return self.behavior_policy.suggest_action_for_state(state, ground_state)

    def update(self, state, action, delta):
        self.target_policy.update(state, action, delta)
        self.behavior_policy.update(state, action, delta)

    def try_initialize_state(self, state, available_actions):

        if self.target_policy.is_new_state(state):
            self.target_policy.initialize_state(state, available_actions)

        if self.behavior_policy.is_new_state(state):
            self.behavior_policy.initialize_state(state, available_actions)

    def learn_episode(self, mdp, step_limit=None, per_step_callback=None):

        self.try_initialize_state(mdp.state, mdp.available_actions)

        while len(mdp.available_actions) > 0:

            if not step_limit is None:
                step_limit -= 1
                if step_limit < 0:
                    break

            current_state = mdp.state
            current_action = self.suggest_action_for_state(mdp.state, mdp.ground_state)


            next_state, next_reward = mdp.transition(current_action)
            current_action = mdp.action_history[-1]

            self.try_initialize_state(next_state, mdp.available_actions)
            self.policy_update_after_step(current_state, current_action,
                                          next_state, next_reward,
                                          mdp)

            if per_step_callback:
                per_step_callback.callback(current_state=current_state, 
                                           current_action=current_action, 
                                           next_state=next_state,
                                           next_reward=next_reward)

        self.policy_update_after_episode(mdp)

    def generate_episode_with_target_policy(self, mdp, step_limit=None, per_step_callback=None):

        self.try_initialize_state(mdp.state, mdp.available_actions)

        while len(mdp.available_actions) > 0:

            if not step_limit is None:
                step_limit -= 1
                if step_limit < 0:
                    break

            current_state = mdp.state
            current_action = self.suggest_action_for_state(mdp.state, mdp.ground_state)

            next_state, next_reward = mdp.transition(current_action)
            self.try_initialize_state(mdp.state, mdp.available_actions)

            if per_step_callback:
                per_step_callback.callback(current_state=current_state, 
                                  current_action=current_action, 
                                  next_state=next_state,
                                  next_reward=next_reward)
