from functools import partial
import os
from pprint import pprint
from os.path import join
from pickle import load, dump
from csv import writer as csv

import torch

from search_strategies import create_search_strategy
from pcfg import PCFG
from grammars import grammars
from network import Network
from evaluation import evaluation_fn
from arguments import parse_arguments
from data import get_data_loaders
from utils import load_config, set_dataset_specific_args, get_exp_path, Limiter
from functools import partial
from main import set_seed


# parse the arguments
args = parse_arguments()
args = load_config(args)
args = set_dataset_specific_args(args)
pprint(vars(args))

save_path = join(args.results_path, get_exp_path(args), "best_architecture.csv")
if not os.path.exists(save_path):
    # set the seed
    set_seed(args.seed)

    # get data loaders
    _, _, trainval_loader, test_loader = get_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        root=args.data_path,
        load_in_gpu=args.load_in_gpu,
        device=args.device,
        log=args.verbose_eval,
        seed=args.seed,
    )

    # get batch for batch pass time limiting
    for batch in trainval_loader:
        batch = batch[0].to(args.device)
        break

    def compile_fn(node, args):
        backbone = node.build(node, set_memory_checkpoint=True)
        return Network(
            backbone,
            node.output_params["shape"],
            args.num_classes,
            vars(args)
        ).to(args.device)

    # create the limiter
    # this makes sure that the search does not exceed
    # time, memory, depth, or node limits during the search
    limiter = Limiter(
        limits={
            "time": args.time_limit,
            "max_id": args.max_id_limit,
            "depth": args.depth_limit,
            "memory": args.mem_limit,
            "individual_memory": args.individual_mem_limit,
            "batch_pass_seconds": args.batch_pass_limit,
        },
        batch=batch,
        compile_fn=partial(compile_fn, args=args),
    )
    limiter.set_memory_checkpoint()
    print(f"Memory checkpoint: {limiter.memory_checkpoint} MB")

    # create the grammar
    grammar = PCFG(
        grammar=grammars[args.search_space],
        limiter=limiter,
    )
    print(grammar)

    args.epochs = args.test_epochs
    eval_fn = partial(
        evaluation_fn,
        args=args,
        train_loader=trainval_loader,
        val_loader=test_loader,
    )

    # load the search results (as a list of tuples: (architecture, reward, duration, timestamp))
    search_results = load(open(join(
        args.results_path,
        get_exp_path(args),
        "search_results.pkl"
    ), "rb"))

    # get the best architecture, sorted by "accuracy"
    # results = sorted(search_results["rewards"][:args.steps], key=lambda x: x[1], reverse=True)[0]
    rewards = search_results["rewards"][:args.steps]
    idx, best = max(enumerate(rewards), key=lambda x: x[1][1])
    best_architecture, best_val_score = best[0], best[1]

    # evaluate the best architecture
    limiter.timer.start()
    test_score = eval_fn(best_architecture[0])
    eval_duration = limiter.timer()
    print(f"ID: {idx}, Test score: {test_score:.4f} (val score of {best_val_score:.4f}) - Time: {eval_duration:.2f}s")

    # save the best architecture and its evaluation in csv format
    with open(save_path, "w") as f:
        csv_writer = csv(f)
        csv_writer.writerow(["Architecture", "Accuracy", "Time"])
        csv_writer.writerow([best_architecture[0], test_score, eval_duration])
else:
    print("Final architecture testing already done.")
