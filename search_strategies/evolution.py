from collections import deque
from copy import deepcopy
import math
from os.path import join, exists
from os import makedirs, rename, remove
import pickle
import random

from tqdm import tqdm

from search_strategies.random_search import Sampler
from baselines import build_baseline, baseline_dict
from visualise import visualise_derivation_tree
from search_strategies.utils import constrained_smith_waterman_crossover
from plot import Plotter

import torch


class Individual(object):
    """A class representing a model containing an architecture, its modules and its accuracy."""

    def __init__(
        self,
        id,
        ancestry=None,
        root=None,
        accuracy=None,
        age=0,
        hpo_dict=None,
    ):
        self.id = id
        self.ancestry = ancestry
        self.root = root
        self.accuracy = accuracy
        self.age = age
        self.hpo_dict = hpo_dict

        self.alive = True

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Individual(id={self.id}, accuracy={self.accuracy}, age={self.age})"

    def __eq__(self, other):
        return self.id == other.id


class Population(deque):
    """A class representing a population of models."""

    def __init__(self, individuals):
        self.individuals = individuals

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Population(individuals={self.individuals})"

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __setitem__(self, idx, value):
        self.individuals[idx] = value

    def append(self, individual):
        self.individuals.append(individual)

    def popleft(self):
        individual = self.individuals.pop(0)
        individual.alive = False

    def without(self, individual):
        return Population([i for i in self.individuals if i != individual])

    def max(self, key):
        return max(self.individuals, key=key)

    def sample(self, k):
        return random.choice(self.individuals, k=k)

    def extend(self, individuals):
        self.individuals.extend(individuals)

    def sort(self, key):
        self.individuals.sort(key=key)

    def __iter__(self):
        return iter(self.individuals)

    def __next__(self):
        return next(self.individuals)

    def __contains__(self, item):
        return item in self.individuals

    def index(self, item):
        return self.individuals.index(item)

    def remove(self, item):
        self.individuals.remove(item)

    def tournament_selection(self, k, key):
        sample = []
        while len(sample) < k:
            candidate = random.choice(list(self.individuals))
            sample.append(candidate)
        return max(sample, key=key)

    def age(self):
        for individual in self.individuals:
            individual.age += 1

    def tolist(self):
        return self.individuals


class Evolver(Sampler):
    def __init__(
        self,
        pcfg=None,
        mode="iterative",
        limiter=None,
        mutation_strategy="random",
        mutation_rate=1.0,
        crossover_strategy="one_point",
        crossover_rate=0.5,
        selection_strategy="tournament",
        tournament_size=10,
        elitism=True,
        verbose=False,
    ):
        super().__init__(
            pcfg=pcfg,
            limiter=limiter,
            mode=mode,
            verbose=verbose
        )
        self.limiter = limiter
        self.mutation_strategy = mutation_strategy
        self.mutation_rate = mutation_rate
        self.crossover_strategy = crossover_strategy
        self.crossover_rate = crossover_rate
        self.selection_strategy = selection_strategy
        self.tournament_size = tournament_size
        self.elitism = elitism

    def evolve(self, population):
        # select the parents (and avoid incest)
        parent1 = self.select(population)
        parent2 = self.select(population.without(parent1))
        # print(f"Parent 1: {parent1.id}, {parent1.accuracy}, {parent1.root}")
        crossover_info = {
            "crossover_strategy": self.crossover_strategy,
            "crossover_rate": self.crossover_rate,
            "parent1": parent1.root, "parent2": None,
            "parent1_id": parent1.id, "parent2_id": None,
            "parent1_accuracy": parent1.accuracy, "parent2_accuracy": None,
        }
        if self.crossover_rate > 0:
            # print(f"Parent 2: {parent2.id}, {parent2.accuracy}, {parent2.root}")
            crossover_info.update({
                "parent2": parent2.root,
                "parent2_id": parent2.id,
                "parent2_accuracy": parent2.accuracy,
            })
        # crossover the parents
        child_root, more_crossover_info = self.crossover(parent1.root, parent2.root)
        crossover_info.update(more_crossover_info)
        # print(f"Child: {child_root}")
        # mutate the child
        child_root, mutation_info = self.mutate(child_root)
        mutation_info.update({
            "mutation_strategy": self.mutation_strategy,
            "mutation_rate": self.mutation_rate,
        })
        # print(f"Mutated child: {child_root}")
        # print("child after mutation")
        # print([(node.operation.name, node.id) for node in child_root.serialise()])
        ancestry = {**crossover_info, **mutation_info}
        return child_root, ancestry

    def select(self, population):
        if self.selection_strategy == "tournament":
            return population.tournament_selection(self.tournament_size, key=lambda x: x.accuracy)
        elif self.selection_strategy == "first":
            return population[0]
        elif self.selection_strategy == "last":
            return population[-1]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            if self.crossover_strategy == "one_point":
                return self.one_point_crossover(parent1, parent2)
            # elif self.crossover_strategy == "two_point":
            #     return self.two_point_crossover(parent1, parent2)
            elif self.crossover_strategy == "constrained_smith_waterman":
                return self.constrained_smith_waterman_crossover(parent1, parent2)
        return parent1, {"crossover": False}

    def one_point_crossover(self, parent1, parent2):
        root_input_params = deepcopy(parent1.input_params)
        successes = [False, False]
        tries = 0
        while not any(successes):
            if tries > 10:
                raise RuntimeError("Crossover failed to generate valid children.")
            # filter valid nodes from parents without copying
            valid_nodes1 = [
                node for node in parent1.serialise()
                if node.operation.type == 'nonterminal'
                and node.parent is not None  # Exclude root nodes
            ]
            valid_nodes2 = [
                node for node in parent2.serialise()
                if node.operation.type == 'nonterminal'
                and node.parent is not None  # Exclude root nodes
            ]

            # ensure valid nodes for swapping
            if not valid_nodes1 or not valid_nodes2:
                raise ValueError("No valid nodes available for crossover.")

            # randomly select nodes
            node1 = random.choice(valid_nodes1)
            node2 = random.choice(valid_nodes2)

            # create deep copies of the parents for the children
            child1_copy = deepcopy(parent1)
            child2_copy = deepcopy(parent2)

            # locate the corresponding nodes in the deep copies
            node1_copy = next(
                node for node in child1_copy.serialise() if node.id == node1.id
            )
            node2_copy = next(
                node for node in child2_copy.serialise() if node.id == node2.id
            )

            # locate parent and index of the copied nodes
            parent1_ref = node1_copy.parent
            idx1 = node1_copy.parent.children.index(node1_copy)

            parent2_ref = node2_copy.parent
            idx2 = node2_copy.parent.children.index(node2_copy)

            # swap children in the deep copies
            parent1_ref.children[idx1] = node2_copy
            parent2_ref.children[idx2] = node1_copy

            # update parent references for swapped nodes
            node1_copy.parent = parent2_ref
            node2_copy.parent = parent1_ref

            # re-infer all params
            try:
                child1.input_params = root_input_params
                child1 = self.re_id(child1_copy)
                successes[0] = True
            except:
                pass
            try:
                child2.input_params = root_input_params
                child2 = self.re_id(child2_copy)
                successes[1] = True
            except:
                pass
            tries += 1

        # TODO FIXME return both children?
        # if successes[0] and successes[1]:
        #     return [child1_individual, child2_individual]
        crossover_info = {
            "crossover": True,
            "crossover_node1_id": node1.id, "crossover_node2_id": node2.id,
            "crossover_node1_depth": node1.depth, "crossover_node2_depth": node2.depth,
            "crossover_node1_operation_type": node1.operation.type, "crossover_node2_operation_type": node2.operation.type,
            "crossover_node1_operation_name": node1.operation.name, "crossover_node2_operation_name": node2.operation.name,
            "crossover_success1": successes[0], "crossover_success2": successes[1],
        }
        if successes[0]:
            return child1, crossover_info
        elif successes[1]:
            return child2, crossover_info


    def two_point_crossover(self, parent1, parent2):
        # TODO Equivalence between one-point and two-point strategies
        # TODO Implement
        return this_is_a_stub

    def constrained_smith_waterman_crossover(self, parent1, parent2, skewness=0, max_tries=100):
        root_input_params = deepcopy(parent1.input_params)
        success = False
        tries = 0
        while not success:
            if tries > max_tries:
                raise RuntimeError("Crossover failed to generate valid children.")
            tries += 1
            child, crossover_operations, crossover_all_operations, distance_to_parent1, distance_to_parent2, distance_between_parents = constrained_smith_waterman_crossover(
                parent1, parent2, skewness=skewness
            )
            # re-infer all params
            try:
                child.input_params = root_input_params
                child = self.re_id(child)
                model = child.build(child)
                model(torch.randn(*self.limiter.batch_shape))
                return child, {
                    "crossover": True, "crossover_operations": crossover_operations,
                    "crossover_distance_to_parent1": distance_to_parent1,
                    "crossover_distance_to_parent2": distance_to_parent2,
                    "crossover_distance_between_parents": distance_between_parents,
                    "crossover_all_operations": crossover_all_operations, "crossover_skewness": skewness,
                }
            except Exception as e:
                print(e)

    def mutate(self, root):
        if random.random() < self.mutation_rate:
            if self.mutation_strategy == "random":
                return self.random_mutation(root, allowed_types=["terminal", "nonterminal"])
            elif self.mutation_strategy == "random_terminal":
                return self.random_mutation(root, allowed_types=["terminal"])
        return root, {"mutation": False}

    def random_mutation(self, root, allowed_types=["terminal", "nonterminal"], max_tries=100):
        success = False
        tries = 0
        while not success:
            if tries > max_tries:
                raise RuntimeError("Mutation failed to generate valid children.")
            tries += 1
            try:
                # print(f"Mutating architecture:")
                root_copy = deepcopy(root)
                # print(f"{root_copy}")
                nodes = root_copy.serialise()
                allowed_nodes = [node for node in nodes if node.operation.type in allowed_types]
                # print(allowed_nodes)
                # choose a random node to mutate
                node = random.choice(allowed_nodes)
                # print(f"Mutating node:")
                # print(f"{node}")
                # mutate the node
                root_mutated = self.mutate_node(root_copy, node)
                root_mutated = self.re_id(root_mutated)
                return root_mutated, {
                    "mutation": True,
                    "mutation_node_id": node.id,
                    "mutation_node_depth": node.depth,
                    "mutation_node_operation_type": node.operation.type,
                    "mutation_node_operation_name": node.operation.name,
                }
            except Exception as e:
                print("MutationError:", e)

    def mutate_node(self, root, node):
        if node.is_leaf():
            # remove the current option from the available options of this node
            if node.available_rules is None:
                options, probs = self.pcfg.get_available_options(node)
                node.available_rules = {
                    "options": options,
                    "probs": probs,
                }
            node.limit_options(node.operation)
            # print(f"Available options: {[op.name for op in node.available_rules['options']]}")
        # sample a new subtree rooted at this node
        self.limiter.timer.start()
        # print(f"Old subtree:")
        # print(f"{node}")
        new_node = self.sample(input_params=node.input_params, root=node)
        # print(f"New subtree:")
        # print(f"{new_node}")
        # replace the old node with the new subtree
        node.replace(new_node)
        # print(f"Mutated architecture:")
        # print(f"{root}")
        # test to see if the new architecture is valid
        # this will run through the entire network with the existing operations
        # and raise an error if the network is invalid
        # print(f"Testing mutated architecture:")
        # print(f"Inputs to sample: {root.input_params}")
        # print(f"Root: {root}")
        # print(f"Operations: {[node.operation.name for node in root.serialise()]}")
        # self.re_id(root)
        # self.limiter.timer.start()
        # self.sample(
        #     input_params=root.input_params,
        #     root=root,
        #     operations=[
        #         node.operation
        #         for node in root.serialise()
        #     ],
        # )
        # print(f"Mutation successful")
        # print(f"New architecture:")
        # print(f"{root}")
        return root


class Evolution:
    def __init__(
            self,
            evaluation_fn,
            pcfg,
            limiter,
            input_params,
            seed=0,
            mode="iterative",
            backtrack=True,
            verbose=False,
            visualise=False,
            visualise_scale=0.5,
            vis_interval=10,
            figures_path=None,
            results_path=None,
            continue_search=False,
            load_from=None,
            # evolution specific parameters
            generational=False,
            regularised=True, # use regularised evolution
            population_size=100, # number of individuals in the population
            architecture_seed=None,
            mutation_strategy="random", # "random"
            mutation_rate=1.0, # probability of mutation
            crossover_strategy="one_point", # "one_point" or "two_point"
            crossover_rate=0.5, # probability of crossover
            selection_strategy="tournament", # "tournament" or "roulette"
            tournament_size=10, # only used if selection_strategy is "tournament"
            elitism=None, # number of best individuals to keep in the population
            n_tries=None, # number of tries to use in evolution before randomly generating an individual
        ):
        self.evaluation_fn = evaluation_fn
        self.pcfg = pcfg
        self.limiter = limiter
        self.input_params = input_params
        self.seed = seed
        self.mode = mode
        self.backtrack = backtrack
        self.verbose = verbose
        self.visualise = visualise
        self.visualise_scale = visualise_scale
        self.vis_interval = vis_interval
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search
        self.load_from = load_from
        # evolution specific parameters
        self.generational = generational
        self.regularised = regularised
        self.population_size = population_size
        self.architecture_seed = architecture_seed
        if self.architecture_seed:
            self.architecture_seed = (
                architecture_seed.split('+') * 
                math.ceil(self.population_size / len(architecture_seed.split('+')))
            )[:self.population_size]
        self.seed_population = {}
        print(f"Architecture seed: {self.architecture_seed}")
        self.elitism = elitism
        self.n_tries = n_tries

        self.evolver = Evolver(
            pcfg=pcfg,
            mode=mode,
            limiter=limiter,
            mutation_strategy=mutation_strategy,
            mutation_rate=mutation_rate,
            crossover_strategy=crossover_strategy,
            crossover_rate=crossover_rate,
            selection_strategy=selection_strategy,
            tournament_size=tournament_size,
            elitism=elitism,
            verbose=verbose,
        )

        self.rewards = []
        self.iteration = 0
        self.population = Population([])
        if self.generational:
            self.old_population = Population([])

        self.set_rng_state(seed=self.seed)

        if self.continue_search:
            self.load_results()

        # fix for the clock
        self.limiter.timer.start()
        print(f"Initialised Evolution at {self.limiter.timer.start_time}")

    def set_rng_state(self, seed=None, state=None):
        if state:
            random.setstate(state)
        elif seed:
            random.seed(seed)

    def learn(self, steps):
        print("-------------")
        print("Evolution")
        print(f"Steps: {steps}")
        print("--------------")

        # populate the first generation
        for iteration in tqdm(range(self.iteration, self.population_size), desc="Initialising population", initial=self.iteration, total=self.population_size):
            if self.architecture_seed:
                self.step(iteration, "seed")
            else:
                self.step(iteration, "sample")

        if self.iteration < self.population_size:
            self.iteration = self.population_size

        for iteration in tqdm(range(self.iteration, steps), desc="Evolving population", initial=self.iteration, total=steps):
            pop_iteration = iteration % self.population_size
            # if starting a new generation, save the old one and open a new empty one
            if self.generational and pop_iteration == 0:
                self.old_population = self.population
                self.population = Population([])
                print(f"New Generation!")
            if self.generational and self.elitism is not None and pop_iteration < self.elitism:
                self.step(iteration, "elite")
            else:
                self.step(iteration, "evolve")

    def step(self, iteration, mode):
        success = False
        n_tries = 0
        while not success:
            n_tries += 1
            try:
                # start timer
                self.limiter.timer.start()

                # sample a new individual
                should_be_random = self.n_tries is not None and n_tries > self.n_tries
                if mode == "seed":
                    seed_arch_name = self.architecture_seed.pop(0)
                    seed_arch = baseline_dict[seed_arch_name]
                    root = build_baseline(seed_arch, self.input_params)
                    ancestry = None
                elif mode == "sample" or should_be_random:
                    root = self.evolver.sample(self.input_params)
                    ancestry = None
                elif mode == "elite":
                    # take best arch from old 
                    pop_iteration = iteration % self.population_size
                    individual = sorted(self.old_population, key=lambda individual: individual.accuracy)[-(1 + pop_iteration)]
                    root, ancestry = individual.root, individual.ancestry
                    print(f"Keeping elite architecture: {individual}")
                elif mode == "evolve":
                    population = self.old_population if self.generational else self.population
                    root, ancestry = self.evolver.evolve(population)
                    print(f"Evolved architecture: {root}")
                sample_duration = self.limiter.timer()

                # check if batch pass does not exceed the time limit
                if not self.limiter.check_batch_pass_time(root, check_memory=True):
                    print("Batch pass time or memory exceeded, trying again")
                    continue

                # start timer
                self.limiter.timer.start()

                # evaluate the network
                if mode == "seed" and seed_arch_name in self.seed_population:
                    print(f"Seed architecture already evaluated: {seed_arch_name}")
                    root, reward, sample_duration, eval_duration = self.seed_population[seed_arch_name]
                else:
                    print(f"Evaluating architecture: {root}")
                    reward = self.evaluation_fn(root)
                    eval_duration = self.limiter.timer()
                    if mode == "seed":
                        self.seed_population[seed_arch_name] = (root, reward, sample_duration, eval_duration)

                success = True
            except (RuntimeError, MemoryError) as e:
                print(f"Error in generating new individual: {e}")

        # add the new individual to the population
        individual = Individual(id=iteration, ancestry=ancestry, root=root, accuracy=reward)
        self.rewards.append((root.serialise(), reward, sample_duration, eval_duration, ancestry))
        self.population.append(individual)
        print(f"Iteration {iteration}, reward: {reward:.2f}, sample duration: {sample_duration:.2f}, eval duration: {eval_duration:.2f}")
        print(f"Architecture: {root}")

        if self.regularised:
            # remove the oldest individual from the population
            if len(self.population) >= self.population_size:
                self.population.popleft()

        self.plot(root, reward, iteration)

        # save the results
        self.save_results(iteration)

    def plot(self, root, reward, iteration):
        # visualise the derivation tree
        root = self.evolver.re_id(root)
        visualise_derivation_tree(
            root,
            scale=self.visualise_scale,
            iteration=iteration,
            save_path=self.figures_path,
            score=reward,
            show=self.visualise,
        )
        if iteration % self.vis_interval == 0:
            plotter = Plotter({"rewards": self.rewards})
            # find best architecture
            idx, best_arch, best_reward = plotter.find_best_architecture()
            # visualise it
            best_root = self.evolver.re_id(best_arch[0])
            visualise_derivation_tree(
                best_root, iteration=f"best_{idx}", score=best_reward, show=False,
                save_path=self.figures_path
            )
            # plot results
            plotter.plot_results("rewards", self.figures_path)
            # plot number of parameters
            plotter.plot_num_params(self.figures_path)
            # plot number of nodes
            plotter.plot_num_nodes(self.figures_path)

    def save_results(self, iteration):
        if self.results_path:
            makedirs(self.results_path, exist_ok=True)
            temp_path = join(self.results_path, f"search_results_temp.pkl")
            final_path = join(self.results_path, f"search_results.pkl")
            try:
                with open(temp_path, "wb") as f:
                    save_data = {
                        "rewards": self.rewards,
                        "iteration": iteration,
                        "population": self.population.tolist(),
                        "rng_state": random.getstate(),
                    }
                    if self.generational:
                        save_data["old_population"] = self.old_population.tolist()
                    pickle.dump(save_data, f)
                rename(temp_path, final_path)
            except KeyboardInterrupt:
                print("Saving interrupted. Partial results saved.")
                if exists(temp_path):
                    remove(temp_path)

    def load_results(self):
        # load the search results
        path = join(self.results_path, "search_results.pkl")
        if not exists(path) and self.load_from:
            path = self.load_from
        if exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.rewards = data["rewards"]
                self.iteration = data["iteration"] + 1
                self.population = Population(data["population"])
                if self.generational:
                    self.old_population = Population(data["old_population"])
                # set the random seed
                self.set_rng_state(state=data["rng_state"])
                print(f"Continuing search from iteration {self.iteration}")
        else:
            print("No previous search results found, starting from scratch")
