# Evolutionary Architecture Search Through Grammar-Based Sequence Alignment

## Experiments & visualizations

To reproduce all experiments, please make sure you have the following files available, that came with this repository:
* `benchmark.pkl`
* `models.pkl`

### Environment

To set up the environment, please create a virtual environment in Python using `venv`, using the given `requirements.txt`.

```
$ pip install -r requirements.txt
```

For CUDA and other specific hardware requirements, please make sure to collect and activate software packages accordingly.
Alternatively, there is also a [nix-shell](https://nixos.wiki/wiki/Development_environment_with_nix-shell) for cross platform support found in `shell.nix`.
Furthermore, a module configuration for [JUWELS](https://www.fz-juelich.de/en/ias/jsc/systems/supercomputers/juwels) is `scripts/juwels_modules.txt`.

### Runtime benchmarks on selected derivation trees

*This will generate Figure 3 of the publication.*

For benchmarking SEPX and CSWX on collected randomly generated trees and derivation trees respectively (cf. `benchmark.pkl` for collected derivation trees during evolutionary search), please collect relevant runtime data first:

```
$ ./benchmark.py --help
$ ./benchmark.py benchmark --runs 10 sepx 1 20
$ ./benchmark.py benchmark --runs 10 cswx1 1 200
```

After having collected data, please run visualizations, to generate plots:

```
$ ./benchmark.py plot
```

For further help, please use the following commands

```
$ ./benchmark.py --help
$ ./benchmark.py benchmark --help
$ ./benchmark.py plot --help
```

### Dataset exploration benchmarks

*This will generate Figure 4 of the publication.*

There are eight datasets given:

* addnist
* language
* multnist
* cifartile
* gutenberg
* isabella
* geoclassing
* chesseract

Please obtain each dataset from their respective source.
After having collected necessary data, experiments can be started within different environments.

#### Requirements

The given benchmarks run ablation studies that require a non-negligible amount of today's compute.
We give a naïve implementation for theoretical reproduction purposes and a configuration specific to JUWELS.

A search space is spanned across multiple configurations, as defined in the file `configs/einspace/evolution_config_list.lst`.
This search space can be iterated upon in a different variety of commands.

#### Naïve implementation

A straight-forward way of running this list is by taking each entry from the given file and feeding it to `scripts/run.sh`.
However, without parallelization this is infeasible.

#### On JUWELS

To queue and evenly distribute the workload, we split the configurations to jobs with 4 configurations each.
Using the main list as a starting point, to then distribute them evenly as individual jobs using SLURM on JUWELS.
Unfortunately the situation with SLURM and paths requires us to use absolute paths:

```
$ split -l 4 configs/einspace/evolution_config_list.lst configs/einspace/evolution_config_list_split4_
for fn in /path/to/einsearch/configs/einspace/evolution_config_list_split4_*; do sbatch -D /path/to/einsearch -A hai_1006 --export ALL --time 06:00:00 --ntasks 1 --gpus-per-task 4 /path/to/einsearch/scripts/slurm-run.sh "$fn" 4; done
```

#### Visualization

After having run the computations above, the `results` directory will contain the runs' relevant collected information.
For visualization the following script can be run:

```
$ ./exploration.py
```

### Cost matrix

*This will generate Figure 2 of the publication.*

The cost matrix and path operations resulting from the application of CSWX can be found in `notebooks/cswx-cost-matrix.ipynb`.


