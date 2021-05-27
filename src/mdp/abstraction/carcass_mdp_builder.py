from . import Carcass

class CarcassBuilder:

    def __init__(self, mdp_builder, rules_filename):
        self.mdp_builder = mdp_builder
        self.rules_filename = rules_filename

    def build_mdp(self):
        ground_mdp = self.mdp_builder.build_mdp()
        return Carcass(ground_mdp, self.rules_filename)

    @property
    def mdp_interface_file_path(self):
        return self.mdp_builder.mdp_interface_file_path


    @property
    def mdp_problem_file_path(self):
        return self.mdp_builder.mdp_problem_file_path


    @property
    def mdp_state_static(self):
        return self.mdp_builder.mdp_state_static


