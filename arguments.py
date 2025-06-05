import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MCTS on a grammar.")
    parser.add_argument("--config", type=str, default="configs/switch_quick/mlp/mcts.yaml", help="The config file to use.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use.")
    # search details
    parser.add_argument("--search_space", type=str, default="einspace", help="The grammar to use.")
    parser.add_argument("--search_strategy", type=str, default="mcts", help="The search strategy to use.")
    parser.add_argument("--steps", type=int, default=1000, help="The number of search steps.")
    parser.add_argument("--backtrack", action="store_true", help="Backtrack when out of options.")
    parser.add_argument("--mode", type=str, default="iterative", help="The mode to use.")
    parser.add_argument("--time_limit", type=int, default=300, help="The time limit to use.")
    parser.add_argument("--max_id_limit", type=int, default=10000, help="The maximum ID limit to use.")
    parser.add_argument("--depth_limit", type=int, default=20, help="The depth limit to use.")
    parser.add_argument("--mem_limit", type=int, default=4096, help="The memory limit in MB to use.")
    parser.add_argument("--individual_mem_limit", type=int, default=1024, help="The memory limit in MB to use.")
    parser.add_argument("--batch_pass_limit", type=int, default=0.1, help="The batch pass limit in seconds to use.")
    # search strategy specific details
    # mcts
    parser.add_argument("--acquisition_fn", type=str, default="uct", help="The acquisition function to use.")
    parser.add_argument("--exploration_weight", type=float, default=1.0, help="The exploration weight to use.")
    parser.add_argument("--incubent_type", type=str, default="parent", help="The incubent type to use in Expected Improvement")
    parser.add_argument("--reward_mode", type=str, default="sum", help="The reward mode to use.")
    parser.add_argument("--add_full_paths", action="store_true", help="Add the full paths to the search tree in MCTS.")
    # evolution
    parser.add_argument("--generational", type=bool, default=False, help="Use generational evolution (as opposed to steady-state).")
    parser.add_argument("--regularised", type=bool, default=False, help="Use regularised evolution.")
    parser.add_argument("--population_size", type=int, default=100, help="The population size to use.")
    parser.add_argument("--architecture_seed", type=str, default=None, help="Baseline architectures to seed the search with, separate with '+', e.g. resnet18+vit+mlpmixer.")
    parser.add_argument("--mutation_strategy", type=str, default="random", help="The mutation strategy to use.")
    parser.add_argument("--mutation_rate", type=float, default=1.0, help="The mutation rate to use.")
    parser.add_argument("--crossover_strategy", type=str, default="random", help="The crossover strategy to use.")
    parser.add_argument("--crossover_rate", type=float, default=0.5, help="The crossover rate to use.")
    parser.add_argument("--selection_strategy", type=str, default="tournament", help="The selection strategy to use.")
    parser.add_argument("--tournament_size", type=int, default=10, help="The tournament size to use.")
    parser.add_argument("--elitism", type=int, default=None, help="Number of elites to keep across generations.")
    parser.add_argument("--n_tries", type=int, default=None, help="The number of tries to use in evolution before randomly generating an individual.")
    # evaluation details
    parser.add_argument("--dataset", type=str, default="mnist", help="The dataset to use.")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use.")
    parser.add_argument("--channels", type=int, default=1, help="The number of channels in the data.")
    # parser.add_argument("--image_size", type=int, default=28, help="The size of the image.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to use.")
    parser.add_argument("--load_in_gpu", action="store_true", help="Load the data in GPU.")
    # logging and plotting
    parser.add_argument("--verbose_search", action="store_true", help="Print verbose output during search.")
    parser.add_argument("--verbose_eval", action="store_true", help="Print verbose output during evaluation.")
    parser.add_argument("--visualise", action="store_true", help="Visualise the derivation tree.")
    parser.add_argument("--visualise_scale", type=float, default=0.8, help="The scale of the visualisation.")
    parser.add_argument("--vis_interval", type=int, default=10, help="The interval to log the results.")
    # load data and saving results and figures
    parser.add_argument("--data_path", type=str, default="data", help="The path to load data from.")
    parser.add_argument("--figures_path", type=str, default="figures", help="The path to save the figures.")
    parser.add_argument("--results_path", type=str, default="results", help="The path to save the results.")
    # continue search
    parser.add_argument("--continue_search", action="store_true", help="Continue the search.")
    parser.add_argument("--load_from", type=str, default=None, help="Path to load search history from.")
    args = parser.parse_args()
    return args
