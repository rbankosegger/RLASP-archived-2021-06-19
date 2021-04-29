import clingo

class Carcass:

    def __init__(self, mdp, rules=[]):

        self.mdp = mdp
        self.rules = rules

        self.available_actions=set()
        self._ground_actions=dict()
        self.state=None

        self._update_abstract_state()

    def _update_abstract_state(self):

        rule_applicable = False

        for rule_id, rule in enumerate(self.rules):

            ctl = clingo.Control()
            ctl.add('base', [], rule)
            ctl.add('base', [], ' '.join(f'{s}.' for s in self.mdp.state))
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
