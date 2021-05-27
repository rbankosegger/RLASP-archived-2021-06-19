import itertools


number_of_repititions = 1

learning_rates = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
epsilons = [0.05, 0.1, 0.2, 0.3]
blocksworld_sizes = [5, 10, 15, 20]

def setup(i, learning_rate, epsilon, blocks_world_size):


    args = [
        f'--db_file=returns.{i:04d}.csv',
        f'--episodes=3000',
        f'--max_episode_length={3*blocks_world_size+1}', 
        f'--control_algorithm=q_learning',
        f'--learning_rate={learning_rate}',
        f'--epsilon={epsilon}',
        f'--no_planning',
        f'--carcass=blocksworld_stackordered.lp',
        f'blocksworld --blocks_world_size={blocks_world_size}'
    ]

    return ' '.join(args)

with open('template.sh', 'r') as templatefile:
    template = templatefile.read()

parameters_list = list(itertools.product(learning_rates, epsilons, blocksworld_sizes))

for i, parameters in enumerate(parameters_list * number_of_repititions):

    with open(f'htcondor_workspace/run.{i:04d}.sh', 'w') as runfile:

        runfile.write(template.format(args_imported_by_python=setup(i, *parameters)))
