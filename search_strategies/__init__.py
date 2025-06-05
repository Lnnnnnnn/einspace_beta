from os.path import join

from .random_search import RandomSearch
from .evolution import Evolution
from .mcts import MCTS
import utils

__all__ = [
    "RandomSearch",
    "Evolution",
    "MCTS",
]


def create_search_strategy(args, grammar, evaluation_fn, limiter, input_params):
    # create the search strategy
    search_strategy = {
        "random_search": RandomSearch,
        "evolution": Evolution,
        "mcts": MCTS,
    }[args.search_strategy]

    # specific parameters for each search strategy
    search_specific_params = {
        "random_search": {
            "figures_path": join(
                args.figures_path,
                utils.get_exp_path(args),
            ),
            "results_path": join(
                args.results_path,
                utils.get_exp_path(args),
            ),
            "load_from": args.load_from,
        },
        "mcts": {
            "figures_path": join(
                args.figures_path,
                utils.get_exp_path(args),
            ),
            "results_path": join(
                args.results_path,
                utils.get_exp_path(args),
            ),
            "load_from": args.load_from,
            "acquisition_fn": args.acquisition_fn,
            "exploration_weight": args.exploration_weight,
            "incubent_type": args.incubent_type,
            "reward_mode": args.reward_mode,
            "add_full_paths": args.add_full_paths,
        },
        "evolution": {
            "figures_path": join(
                args.figures_path,
                utils.get_exp_path(args),
            ),
            "results_path": join(
                args.results_path,
                utils.get_exp_path(args),
            ),
            "load_from": args.load_from,
            "generational": args.generational,
            "regularised": args.regularised,
            "population_size": args.population_size,
            "architecture_seed": args.architecture_seed,
            "mutation_strategy": args.mutation_strategy,
            "mutation_rate": args.mutation_rate,
            "crossover_strategy": args.crossover_strategy,
            "crossover_rate": args.crossover_rate,
            "selection_strategy": args.selection_strategy,
            "tournament_size": args.tournament_size,
            "elitism": args.elitism,
            "n_tries": args.n_tries,
        }
    }

    # create the search
    search = search_strategy(
        # common parameters
        evaluation_fn=evaluation_fn,
        pcfg=grammar,
        limiter=limiter,
        input_params=input_params,
        seed=args.seed,
        mode=args.mode,
        backtrack=args.backtrack,
        verbose=args.verbose_search,
        visualise=args.visualise,
        visualise_scale=args.visualise_scale,
        vis_interval=args.vis_interval,
        continue_search=args.continue_search,
        # specific parameters
        **search_specific_params[args.search_strategy],
    )

    return search
