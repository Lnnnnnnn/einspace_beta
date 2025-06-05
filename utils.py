import math
from os.path import join
from functools import reduce
from time import perf_counter as time
from pathlib import Path
import psutil
from scipy import stats
import yaml

import sys
import gc
from pympler import asizeof, summary

import io
import torch
from pickle import Unpickler


class CPU_Unpickler(Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def set_dataset_specific_args(args):
    if args.dataset in ["cifar10", "cifar100"]:
        # training details
        args.score = "xe"
        args.epochs = 25
        args.patience = 25
        args.test_epochs = 200
        args.batch_size = 128
        args.hpo_runs = 1
        args.lr = 0.1
        args.momentum = 0.9
        args.weight_decay = 0.0005
        # dataset details
        args.load_in_gpu = False
        args.channels = 3
        args.image_size = [32, 32]
        args.input_mode = "im"
    elif args.dataset in ["addnist", "language", "multnist", "cifartile", "gutenberg", "isabella", "geoclassing", "chesseract"]:
        # training details
        args.score = "xe"
        args.epochs = 8
        args.patience = 8
        args.test_epochs = 64
        args.batch_size = 256
        args.hpo_runs = 1
        args.lr = 0.04
        args.momentum = 0.9
        args.weight_decay = 0.0003

    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "addnist":
        args.load_in_gpu = True
        args.num_classes = 20
        args.channels = 3
        args.image_size = [28, 28]
        args.input_mode = "im"
    elif args.dataset == "language":
        args.load_in_gpu = True
        args.num_classes = 10
        args.channels = 1
        args.image_size = [24, 24]
        args.input_mode = "im"
    elif args.dataset == "multnist":
        args.load_in_gpu = True
        args.num_classes = 10
        args.channels = 3
        args.image_size = [28, 28]
        args.input_mode = "im"
    elif args.dataset == "cifartile":
        args.load_in_gpu = True
        args.num_classes = 4
        args.channels = 3
        args.image_size = [64, 64]
        args.input_mode = "im"
    elif args.dataset == "gutenberg":
        args.load_in_gpu = True
        args.num_classes = 6
        args.channels = 1
        args.image_size = [27, 18]
        args.input_mode = "im"
    elif args.dataset == "isabella":
        args.load_in_gpu = True
        args.num_classes = 4
        args.channels = 1
        args.image_size = [64, 128]
        args.input_mode = "im"
    elif args.dataset == "geoclassing":
        args.load_in_gpu = True
        args.num_classes = 10
        args.channels = 3
        args.image_size = [64, 64]
        args.input_mode = "im"
    elif args.dataset == "chesseract":
        args.load_in_gpu = True
        args.num_classes = 3
        args.channels = 12
        args.image_size = [18, 18]
        args.input_mode = "im"
    elif args.dataset == "mnist":
        args.image_size = [28,28]
        args.channels = 1
        args.num_classes = 10

    # my PHC dataset
    elif args.dataset == "switch15to6":
        args.channels = 15     # input vector length
        args.image_size = []   # not used for vectors
        args.num_classes = 6


    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return args


def load_config(args):
    # load yaml file and overwrite anything in it
    # relative to working dir
    local_config = Path(args.config)
    # relative to script's dir
    relative_config = Path(__file__).resolve().parent / args.config

    print(local_config)
    print(relative_config)

    config = None
    if local_config.exists():
        config = local_config
    if relative_config.exists():
        config = relative_config

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        for key, value in config.items():
            if value == "None":
                config[key] = None
    # convert to args
    for key, value in config.items():
        setattr(args, key, value)
    # ensure device is set
    if args.device is None:
        raise ValueError("Please specify device")
    return args


def get_exp_path(args):
    exp_path = join(
        args.search_space,
        args.dataset,
        args.search_strategy,
        f"seed={args.seed}",
        f"backtrack={args.backtrack}",
        f"mode={args.mode}",
        f"time_limit={args.time_limit}",
        f"max_id_limit={args.max_id_limit}",
        f"depth_limit={args.depth_limit}",
        f"mem_limit={args.mem_limit}",
        f"load_from={args.load_from}",
    )
    if args.search_strategy == "random_search":
        pass
    elif args.search_strategy == "mcts":
        exp_path = join(exp_path, f"acquisition_fn={args.acquisition_fn}")
        if args.acquisition_fn == "uct":
            exp_path = join(exp_path, f"exploration_weight={args.exploration_weight}")
        else:
            exp_path = join(exp_path, f"incubent_type={args.incubent_type}")
        exp_path = join(exp_path, f"reward_mode={args.reward_mode}")
        exp_path = join(exp_path, f"add_full_paths={args.add_full_paths}")
    elif args.search_strategy == "evolution":
        exp_path = join(exp_path, f"generational={args.generational}")
        exp_path = join(exp_path, f"regularised={args.regularised}")
        exp_path = join(exp_path, f"population_size={args.population_size}")
        exp_path = join(exp_path, f"architecture_seed={args.architecture_seed}")
        exp_path = join(exp_path, f"mutation_strategy={args.mutation_strategy}")
        exp_path = join(exp_path, f"mutation_rate={args.mutation_rate}")
        exp_path = join(exp_path, f"crossover_strategy={args.crossover_strategy}")
        exp_path = join(exp_path, f"crossover_rate={args.crossover_rate}")
        exp_path = join(exp_path, f"selection_strategy={args.selection_strategy}")
        exp_path = join(exp_path, f"tournament_size={args.tournament_size}")
        exp_path = join(exp_path, f"elitism={args.elitism}")
    return exp_path


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time()
        # print(f"Timer started at {self.start_time}")

    def stop(self):
        self.end_time = time()

    def __call__(self):
        current_time = time()
        duration = current_time - self.start_time
        # print(f"Timer called at {current_time}, start time: {self.start_time}, duration: {duration}")
        return duration

    def __str__(self):
        return f"Timer(start_time={self.start_time:.2f}, duration={self():.2f})" 


class Limiter:
    def __init__(self, limits, batch=None, compile_fn=None, n_batch_passes=5):
        self.limits = limits
        self.timer = Timer()
        self.memory_checkpoint = None

        if batch is not None:
            self.batch_shape = batch.shape
            self.device = batch.device
        else:
            self.batch_shape = None
            self.device = None
        self.compile_fn = compile_fn
        self.n_batch_passes = n_batch_passes

        print(f"Limiter({self.limits})")

    def check(self, node, verbose=False):
        """
        Check if the limits have been reached.
        """
        # get the current duration
        duration = self.timer()
        # get the memory usage
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB

        # check if the diff between memory and memory checkpoint is over the limit
        self.diff = (self.memory - self.memory_checkpoint) if self.memory_checkpoint is not None else 0
        if (
            node.depth >= self.limits["depth"] or
            duration >= self.limits["time"] or
            node.id >= self.limits["max_id"] or
            self.memory >= self.limits["memory"] or 
            self.diff >= self.limits["individual_memory"]
        ):
            # print which limit was reached
            if node.depth >= self.limits["depth"]:
                limit_reached = "Depth"
            if duration >= self.limits["time"]:
                limit_reached = "Time"
            if node.id >= self.limits["max_id"]:
                limit_reached = "Max_id"
            if self.memory >= self.limits["memory"]:
                limit_reached = "Memory"
            if self.diff >= self.limits["individual_memory"]:
                limit_reached = "Individual Memory"
            # print(f"{limit_reached} limit reached for node {node.id}")
            # print(f"{self}")
            return False
        else:
            return True

    def set_memory_checkpoint(self):
        self.memory_checkpoint = psutil.Process().memory_info().rss / (1024 * 1024)

    def reset_memory_checkpoint(self):
        self.memory_checkpoint = None

    def check_memory(self):
        """
        Check if the limits have been reached.
        """
        # get the memory usage
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        self.diff = (self.memory - self.memory_checkpoint) if self.memory_checkpoint is not None else 0
        if self.diff >= self.limits["individual_memory"]:
            return False
        if self.memory >= self.limits["memory"]:
            return False
        return True

    def check_batch_pass_time(self, node, check_memory=False):
        """
        Check if the limits have been reached.
        """
        if self.batch_shape is None:
            assert self.compile_fn is not None, "Compile function must be provided if limiting batch pass time"
            return True

        if check_memory:
            self.set_memory_checkpoint()

        # node to torch model
        model = self.compile_fn(node)

        # return false is the model is too large
        if check_memory:
            memcheck = self.check_memory()
            # self.reset_memory_checkpoint()
            if not memcheck:
                print(f"Model too large - {self.diff} MB")
                return False

        dur = 0
        timer = Timer()
        for _ in range(self.n_batch_passes):
            timer.start()
            # print(f"Batch pass {node.id}")
            # print(f"check_batch_pass_time with shape: {self.batch_shape}")
            model(torch.randn(self.batch_shape).to(self.device))
            dur += timer()
            # print(f"Batch pass {node.id} completed. {dur} seconds elapsed.")

        duration = dur / self.n_batch_passes

        print(f"Batch pass duration: {duration}")

        if duration >= self.limits["batch_pass_seconds"]:
            return False
        return True
    
    def check_build_safe(self, node):
        # compute the size of extra parameters
        param_size = node.input_params["num_params"] if "num_params" in node.input_params else 0
        # compute size of output tensor
        output_size = reduce(lambda x, y: x*y, node.output_params['shape']) * node.output_params["branching_factor"]
        # combine
        memory_size = param_size + output_size
        # print(f"Additional memory added by operation {node.operation.name}: {memory_size * 4 / (1024 * 1024):.2f} MB")
        return memory_size * 4 / (1024 * 1024) < self.limits["individual_memory"]

    # Function to summarize memory usage of all objects
    def summarise_memory(self, k=3):
        from search_state import DerivationTreeNode, Operation, Stack
        from search_strategies.mcts import SearchTreeNode
        from torch import Size

        print("---- Memory Usage Summary ----")
        # Using gc to inspect all objects
        objects = gc.get_objects()
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        print(f"Memory usage: {self.memory:.2f} MB")
        print(f"Objects tracked by garbage collector: {len(objects)}")
        
        # Display memory usage for each object type
        obj_types = {}
        for obj in objects:
            obj_type = type(obj).__name__
            obj_size = sys.getsizeof(obj)
            obj_types[obj_type] = obj_types.get(obj_type, 0) + obj_size
        
        # print("---- Memory by Object Type ----")
        for i, (obj_type, total_size) in enumerate(sorted(obj_types.items(), key=lambda x: -x[1])):
            print(f"{obj_type}: {total_size / (1024**2):.2f} MB")
            # self.analyse_instances(eval(obj_type))
            if i >= k:
                break
        # self.analyse_instances(DerivationTreeNode)
        # self.analyse_instances(SearchTreeNode)
        
        # print("\n---- Detailed Memory Summary (Pympler) ----")
        # # Generate a more detailed report using pympler
        # memory_summary = summary.summarize(objects)
        # summary.print_(memory_summary)

        # print("---- End of Memory Summary ----")

    def analyse_instances(self, obj_type):
        # Find all instances of the given object type
        instances = [obj for obj in gc.get_objects() if isinstance(obj, obj_type)]
        print(f"Found {len(instances)} instances of {obj_type.__name__}")
        
        # Analyze the memory usage of the first few instances
        for idx, instance in enumerate(instances[:5]):  # Limit to first 5 instances
            print(f"\nInstance {idx+1} of {obj_type.__name__}:")
            print(f"  Total size (recursive): {asizeof.asizeof(instance) / (1024**2):.2f} MB")
            for attr_name in dir(instance):
                if not attr_name.startswith("__"):
                    attr_value = getattr(instance, attr_name)
                    attr_size = asizeof.asizeof(attr_value)
                    print(f"  {attr_name}: {type(attr_value).__name__}, Size: {attr_size / (1024**2):.2f} MB")

    def __str__(self):
        repr = f"Limiter(\n"
        repr += f"\t{self.limits},\n"
        repr += f"\t{self.timer}\n"
        repr += f"\t{self.memory_checkpoint:.2f} MB (baseline memory)\n"
        repr += f"\t{self.memory:.2f} MB (total memory)\n"
        repr += f"\t{self.memory - self.memory_checkpoint:.2f} MB (individual memory)\n"
        repr += ")"
        return repr


def kendall_rank_correlation(all_labels, all_preds):
    """Gets the kendall's tau-b rank correlation coefficient.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    Parameters
    ----------
    all_labels: list
        A list of labels.
    all_preds: list
        A list of predicted values.
    Returns
    -------
    correlation: float
        The tau statistic.
    pvalue: float
        The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
    """

    tau, p_value = stats.kendalltau(all_preds, all_labels)
    return tau


def millify(n, bytes=False, return_float=False):
    n = float(n)
    if bytes:
        millnames = ["B", "KB", "MB", "GB", "TB", "PB"]
    else:
        millnames = ["", "K", "M", "B", "T"]
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    if return_float:
        return n / 10 ** (3 * millidx)
    else:
        return f"{int(n / 10 ** (3 * millidx))}{millnames[millidx]}"


# --------- functions for scanning making predictions from one-hot or multi-hot models
def scan_thresholded(thresh_row):
    predicted_label = 0
    for ind in range(thresh_row.shape[0]):  # start scanning from left to right
        if thresh_row[ind] == 1:
            predicted_label += 1
        else:  # break the first time we see 0
            break
    return predicted_label


def logits_to_preds(logits, loss_type):
    with torch.no_grad():
        if loss_type == 'multi_hot':

            # Apply 0.5 for the 0 or 1
            probs = torch.sigmoid(logits)
            preds_tensor = (probs > 0.5).float()
            preds = preds_tensor.cpu().tolist()

            # probs = torch.sigmoid(logits)
            # thresholded = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))  # apply threshold 0.5
            # preds = []
            # batch_size = thresholded.shape[0]
            # for i in range(batch_size):  # for each item in batch
            #     thresholded_row = thresholded[i, :]  # apply threshold to probabilities to replace floats with either 1's or 0's
            #     predicted_label = scan_thresholded(thresholded_row)  # scan from left to right and make the final prediction
            #     preds.append(predicted_label)

        else:  # softmax followed by argmax
            probs = torch.softmax(logits, dim=1)
            preds_tensor = torch.argmax(probs, dim=1)  # argmax in dim 1 over 8 classes
            preds = [pred.item() for pred in preds_tensor]
        return preds, probs  # preds is 1d list, probs 2d tensor
