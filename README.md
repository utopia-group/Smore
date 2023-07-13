# Code Artifact for Smore

## Overview

This repository contains the code artifact for evaluation section of the paper "Data Extraction via Semantic Regular Expression Synthesis". In particular, it supports the following claims in the evaluation section:

1. Smore outperforms existing automated data extraction baselines in terms of $F_1$ score and the number of tasks finished in 1 minute. (Section 7.1)
2. The neural-guided synthesis technique presented in the paper outperforms existing synthesis techniques. (Section 7.2)
3. Different components in our proposed synthesis algorithms are important for the performance of the overall system. (Section 7.3)

This artifact supports these claims by including the code, benchmarks, and running scripts for reproducing the results.

This artifact does not support the claim that semantic regexes help human more effectively solve data extraction since this requires running a user study. 

## Getting Started Guide

In this section, we provide two ways to set up the artifact: through a docker image (recommended) and through the source code.

### Building and Running the Docker Image

Building the docker image takes around 20 minutes. The recommendation setup is to allocate at least 16G of memory and 6 CPU cores to the docker image.

```
docker build -t smore:v1 .
docker run -it smore:v1 bash
```

### Running from the Source Code

#### System Requirements

This code base is tested on macOS Ventura 13.2.1 with an Apple M2 Max chip and 64GB memory.

#### Setup

This repo requires python version `3.10` and the conda environment management system (note that conda is not required if you are running from the docker image). It also requires installing the following dependencies.

```
curl
gcc
clang
python3-dev
dotnet 5.0 (not required for running the tool, but required for running the flashgpt baseline used in the evaluation)
```

Run the following command to create a new conda environment and activate it (assuming conda is already installed). 

```
conda create --name smore python=3.10
conda activate smore
conda install pip
```

Run the following command to install relevant packages and models:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

To reproduce the evaluation results, unzip the cache file `cache_ae.zip` under the root directory of this repo.

**To run the tool with new tasks**, an OpenAI key is required. This repository uses the following model: `text-davinci-003`, `code-davinci-002` and `gpt-3.5-turbo`. To use these models, you need to set the following environment variables:

```
export OPENAI_API_KEY={your key}
```  

### Verifying the installation

To verify the installation, run the following command:

```
python3 -W ignore test_dsl.py
python3 -W ignore test_overall_synthesizer.py
```

Both of these program executions should finish without any error.

## Step-by-Step Instructions

### Reproducing Claim #1 (Section 7.1)

Recall that Section 7.1 uses the following baseline systems for comparison:

- `ChatGPT-Regex-Synth`: Synthesizing traditional regular expressions using ChatGPT (`gpt-3.5-turbo`)
- `ChatGPT-Exec`: Directly invoke ChatGPT to test the test examples (without synthesizing a program) by providing the training examples as in-context examples
- `FlashGPT`: Synthesizing programs in `FlashGPT` DSL

Detailed descriptions of these baselines can be found in Section 7.1 of the paper.

To run the baselines, run the following command in the following subsections. Each script stdout should end with information in the following template:

```
save_dict_to_csv: {filename}
record len: 50
====================
Final results for {baseline_name} are:
F1 score: {average_f1_score}
Timeout count: {timeout count}
====================
```

The detailed results are saved in the file `filename` under the root directory of this repo.

Once all baselines finished, obtained a summary table by running the command:

```
python3 process_eval_data.py --mode 1
```


#### ChatGPT-Regex-Synth
```
python3 -W ignore run_eval.py --chatgpt_regex --eval_worksheet_name ae_results
```
The stdout output should end with the following:
```
====================
Final results for GPT-3.5-regex- are:
F1 score: 0.43992272739748844
Timeout count: 27
====================
```
Note that the timeout count corresponds to the number of tasks that cannot find a regex by ChatGPT.

#### ChatGPT-Exec
```
python3 -W ignore run_eval.py --chatgpt_exec --eval_worksheet_name ae_results
```
The stdout output should end with the following:
```
====================
Final results for GPT-3.5-exec- are:
F1 score: 0.6477213880565675
Timeout count: 0
====================

```
#### FlashGPT

**Note that this baseline cannot be run through docker** at this point (although all the necessary code are included in this repository).
This baseline requires `dotnet 5.0` to run and takes $2-3$ hours to finish. After installing all the necessary dependencies, run the baseline using the following command:

```
python3 -W ignore run_eval.py --flashgpt_run --eval_worksheet_name ae_results
```

Note that this baseline can easily trigger "Out of Memory" error and takes $2-3$ hours to finish, so we recommend running it on a machine with at least 128GB memory.

We provide the full results in the under the directory `eval_res/flashgpt_results`. Run the following command to post-process the results:

```
python3 -W ignore run_eval.py --flashgpt_postprocess --eval_worksheet_name ae_results
```


#### Smore
```
python3 -W ignore run_eval.py --smore --eval_worksheet_name ae_results
```

The script terminates once it finishes evaluating all the tasks (and it should terminate within 10 minutes). The stdout output should end with the following:

```
====================
Final results for Smore- are:
F1 score: 0.8758067223649199
Timeout count: 3
====================
```
Note that the timeout count is 1 higher than the reported value in the paper (this number might be higher depending on the machine), and it is due to hardware performance restrictions.

### Reproducing Claim #2 (Section 7.2)

Recall that Section 7.2 uses the following baseline systems for comparison:

- `ChatGPT-Synth`: Synthesizing semantic regular expressions using ChatGPT (`gpt-3.5-turbo`)
- `Smore-NoSketch:`: Synthesizing semantic regular expression using Smore without sketch

Detailed descriptions of these baselines can be found in Section 7.2 of the paper.

To run the baselines, run the following command in the following subsections. Once finished, obtained a summary table by running the command:

```
python3 process_eval_data.py --mode 2
```

#### ChatGPT-Synth
```
python3 -W ignore run_eval.py --chatgpt_synth --eval_worksheet_name ae_results
```
The output should end with the following:
```
====================
Final results for GPT-3.5-synth- are:
F1 score: 0.71096934254829
Timeout count: 44
====================
```

#### Smore-NoSketch
```
python3 -W ignore run_eval.py --no_sketch --eval_worksheet_name ae_results
```

The output should end with the following (it takes around 30 minutes for the script to terminate):
```
====================
Final results for Smore-no-sketch- are:
F1 score: 0.7971823889299064
Timeout count: 37
====================
```

### Reproducing Claim #3 (Section 7.3)

Recall that Section 7.3 uses the following variants of Smore for comparison:

- `Smore-NoDecomp`:  A variant of Smore that does not perform compositional sketch completion
- `Smore-NoTypedHole`: A variant of Smore that does not use typed sketches.
- `Smore-NoLocateError`: A variant of Smore that does not perform error localization for sketch
- `Smore-NoTypeSystem`: A variant of Smore that does not perform type-directed synthesis

To run these variants, run the following command in the following subsections.Once finished, obtained a summary table by running the command:

```
python3 process_eval_data.py --mode 3
```

#### Smore-NoDecomp
```
python3 -W ignore run_eval.py --no_decomp --eval_worksheet_name ae_results
```
The output should end with the following (it takes around 30 minutes for the script to terminate):
```
====================
Final results for Smore-no-decomp- are:
F1 score: 0.9286701602491076
Timeout count: 40
====================
```

#### Smore-NoTypedHole
```
python3 -W ignore run_eval.py --no_type --eval_worksheet_name ae_results
```

The output should end with the following (it takes around 25 minutes for the script to terminate):
```
====================
Final results for Smore-no-type- are:
F1 score: 0.7432200539945781
Timeout count: 17
====================
```

#### Smore-NoLocateError
```
python3 -W ignore run_eval.py --no_repair --eval_worksheet_name ae_results
```

The output should end with the following (it takes around 15 minutes for the script to terminate):
```
====================
Final results for Smore-no-repair- are:
F1 score: 0.8819025244334842
Timeout count: 22
====================
```

#### Smore-NoTypeSystem
```
python3 -W ignore run_eval.py --no_type_system --eval_worksheet_name ae_results
```

The output should end with the following (it takes around 30 minutes for the script to terminate):
```
====================
Final results for Smore-no-type-system- are:
F1 score: 0.8590821489125432
Timeout count: 27
====================
```

## Description of the Codebase

The code base is divided into 2 main functionalities: DSL and synthesizer.

The following modules are related to DSL:

- [lang](./lib/lang/): defines the CFG for the SemRegex DSL
- [paser](./lib/parser/): a lark parser that parses from program in string form to a program that can understand by the interpreter. Note that for usability the grammar inside parser.py is slightly different from the grammar in lang. The functions in parser.py helps to convert user-written programs into canonized forms.
- [interpreter](./lib/interpreter/): the actual interpreter of the language.
- [nlp](./lib/nlp/): the execution engine that responsible for executing all nlp-related functions

The following modules are related to the synthesizer:

- [type](./lib/type/): the type system is used to guide the search. Base types currently supported are enumerated in type_enum.py. The typing rules are written in type_system.py.
- [program](./lib/program/): necessary data structures to represent a program as an abstract syntax tree.
- [synthesizer](./lib/synthesizer/):
  - top_level_synthesizer: XXX 
  - sketch_synthesizer: synthesize program given a parsed sketch.
  - decompose: do decomposition based on the generated sketch
  - typed_synthesizer: this file implemented a type-directed synthesized to instantiate each decomposed goal.

To make everything work we also need some utility packages:

- [cache](./lib/cache/): A framework for caching intermedia results for interpreter execution
- [eval](./lib/eval/): contains a wrapper for an evaluation benchmark
- [config](./lib/config/): some high-level global configuration.
- [utils](./lib/utils/): a variety of utility functions. The name should be straight-forward enough to indicate where these utility functions serve at.

### DSL executions

To execute a program on a set of positive and negative examples, refer to [tests/test_dsl.py](./tests/test_dsl.py) on how to instantiate and run the executor class.

In `test_dsl.py`, you will see two ways to create a program:

1. directly calling functions in the interpreter.pattern
2. write the program as a string and call parse_program to parse the program

Thew automated scripts to obtain the current evaluation results w.r.t the manually-written program on collected benchmarks is the file [eval_dsl_accuracy.py](lib/eval/eval_dsl_accuracy.py)
The accuracy should be printed once the execution finished.

### Synthesizer executions

Each part of the synthesizer can be tested individually using the following scripts:
- overall synthesis algorithm: [test_overall_synthesizer.py](./test_overall_synthesizer.py)
- sketch generation: [test_sketch_synthesizer.py](./test_sketch_synthesizer.py)
- type-directed synthesis: [test_type_directed_synthesis.py](./test_type_directed_synthesis.py)
