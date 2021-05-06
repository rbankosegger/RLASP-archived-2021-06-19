import os
import random
import clingo

from .. import StateHistory

class Carcass(StateHistory):

    @staticmethod
    def file_path(file_name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carcass_rules', file_name)

    def __init__(self, mdp, rules_filename):

        self.mdp = mdp
        self.rules_filename = rules_filename 

        self.available_actions=set()
        self._ground_actions=dict()
        self.state=None

        # Set the first abstract state
        self._update_abstract_state()

        super().__init__(self.state)

    def _update_abstract_state(self):

        ctl = clingo.Control()
        ctl.load(self.file_path(self.rules_filename))
        ctl.add('base', [], ' '.join(f'{s}.' for s in self.mdp.state))
        ctl.add('base', [], ' '.join(f'{s}.' for s in self.mdp.state_static))
        ctl.ground(parts=[('base', [])])

        solvehandle = ctl.solve(yield_=True)

        model = solvehandle.model()

    
        #self.state=f'abstract{rule_id}'
        self.available_actions = set()
        self._ground_actions=dict()
        
        for symbol in model.symbols(shown=True):

            if symbol.name == 'choose':

                self.state=f'carcass_{symbol.arguments[0]}'
            
        
            if symbol.name == 'abstractAction':
                
                abstract_action =  str(symbol.arguments[0])
                ground_action = str(symbol.arguments[1])
        
                self.available_actions.add(abstract_action)
        
                self._ground_actions[abstract_action] = self._ground_actions.get(abstract_action, set()) | {ground_action}


        if self.state == 'carcass_gutter':
            
            # No rules are applicable. Group all available ground actions into one single "gutter" action.

            if len(self.mdp.available_actions) > 0:
                self.available_actions = {'random'}
                self._ground_actions = { 'random' : self.mdp.available_actions }

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
