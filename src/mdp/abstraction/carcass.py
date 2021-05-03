import random
import clingo

from .. import StateHistory

class Carcass(StateHistory):

    def __init__(self, mdp, rules=[], background_knowledge=""):

        self.mdp = mdp
        self.rules = rules
        self.background_knowledge = background_knowledge

        self.available_actions=set()
        self._ground_actions=dict()
        self.state=None

        # Set the first abstract state
        self._update_abstract_state()

        super().__init__(self.state)

    def _update_abstract_state(self):

        rule_applicable = False

        for rule_id, rule in enumerate(self.rules):

            ctl = clingo.Control()
            ctl.add('base', [], self.background_knowledge)
            ctl.add('base', [], rule)
            ctl.add('base', [], ' '.join(f'{s}.' for s in self.mdp.state))
            ctl.add('base', [], ' '.join(f'{s}.' for s in self.mdp.state_static))
            ctl.ground(parts=[('base', [])])

            solvehandle = ctl.solve(yield_=True)

            model = solvehandle.model()
            rule_applicable = solvehandle.get().satisfiable

            if rule_applicable:
    
                self.state=f'abstract{rule_id}'
                self.available_actions = set()
                self._ground_actions=dict()
        
                for symbol in model.symbols(shown=True):
        
                    if symbol.name == 'abstractAction':
                        
                        abstract_action =  str(symbol.arguments[0])
                        ground_action = str(symbol.arguments[1])
        
                        self.available_actions.add(abstract_action)
        
                        self._ground_actions[abstract_action] = self._ground_actions.get(abstract_action, set()) | {ground_action}

                break

        if not rule_applicable:
            
            # No rules are applicable. Group all available ground actions into one single "gutter" action.

            self.state = 'gutter'

            if len(self.mdp.available_actions) > 0:
                self.available_actions = {'gutter'}
                self._ground_actions = { 'gutter' : self.mdp.available_actions }

            else:
                self.available_actions = set()
                self._ground_actions= dict()


    def ground_actions_of(self, abstract_action):

        return self._ground_actions.get(abstract_action, dict())

    def transition(self, abstract_action):
    
        ground_action = random.choice(list(self.ground_actions_of(abstract_action)))

        next_ground_state, next_reward = self.mdp.transition(ground_action)

        self._update_abstract_state()

        super().transition(abstract_action, #A[t]
                           self.state, # S[t+1]
                           next_reward # R[t+1]
                          )

        return self.state, next_reward

    @property 
    def discount_rate(self):
        return self.mdp.discount_rate
