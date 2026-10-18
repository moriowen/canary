# Canary

Early warning for coding-agent runs that are going to fail.

An AI coding agent gets a real GitHub issue and works in a loop: it thinks, runs a command, reads the output, and tries again. After a median of 31 messages it submits a patch, and the project's tests decide whether the patch works. In the public SWE-agent dataset, 5 out of 6 runs fail, and most failed runs use the full budget before anyone finds out.

Canary asks whether the first few turns of a run already show where it is heading. If they do, a doomed run can be stopped early and its compute handed to a run that might succeed.

## The data

[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (CC-BY-4.0). These are open-weight Llama models running SWE-agent on local hardware. Every number below comes from `canary metadata`, not from the dataset card.

- 80,036 runs over 3,591 GitHub issues, about 22 attempts per issue
- 13,389 solved (16.7%), 66,647 failed
- Median run length is 31 messages
- Models: llama-70b 74,792, llama-8b 4,053, llama-405b 1,191

How runs end, and how often each ending is a success:

| exit_status | failed | solved | solve rate |
| --- | ---: | ---: | ---: |
| submitted | 38,415 | 12,672 | 24.8% |
| submitted (exit_context) | 20,314 | 712 | 3.4% |
| exit_context | 3,568 | 0 | 0% |
| early_exit | 3,176 | 0 | 0% |
| submitted_no_patch | 1,066 | 0 | 0% |
| others | 108 | 5 | n/a |

59,880 runs (74.8% of the dataset) exited cleanly and still failed. For all of them `exit_status` just says "submitted", so it tells you nothing about what went wrong.

### The confound

2,753 of the 3,591 issues (76.7%) are never solved in any attempt, and only 31 are solved every time. A model can look good by learning which issues are hopeless, without learning anything about how a particular run is going. That model would be useless for stopping runs: it would make its call at step 0 and never change its mind.

The same data offers a control. 807 issues have both outcomes across 34,922 runs, so the task is held fixed and only the run varies. Canary reports results on the full dataset and on this mixed-outcome subset. The gap between the two is the size of the confound.

## What Canary is building

1. Early-abort prediction. From the first *k* turns, predict whether the run will succeed. The headline metric is compute saved at a fixed success rate: how many turns you get back if you are allowed to kill at most a set fraction of runs that would have succeeded.
2. A failure taxonomy. Cluster the 59,880 quiet failures to find out how they fail (never running tests, editing the wrong file, repeating the same command, patching a symptom), since nothing in the dataset currently says.
3. Within-task evaluation on the 807 mixed-outcome issues, as described above.
4. Cross-scaffold transfer. Check whether a predictor trained on SWE-agent runs still works on runs from a different agent framework, such as SWE-smith.

Any model has to beat two simple rules first: abort after 3 consecutive errors, and abort after N turns. If it can't, there is no result.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Usage

```bash
canary metadata            # label columns for all 80k runs + stats (seconds, a few MB)
canary sample              # 6,000-run stratified sample with transcripts (~1.1 GB download)
canary peek --steps 5      # print a failed run as a readable transcript
canary peek --solved       # a successful one, for contrast
```

Data lands in `data/`, which is git-ignored. Run `canary <command> --help` for options.

## Data schema

One row per run:

| Column | Meaning |
| --- | --- |
| `instance_id` | which GitHub issue. Group train/test splits on this. |
| `model_name` | backbone model |
| `target` | bool, whether the patch fixed the issue. This is the label. |
| `trajectory` | list of steps, each with `role` (system, ai, user) and `text` |
| `exit_status` | how the run ended. Known only after the fact, so never a feature. |
| `generated_patch` | the code the agent produced |
| `eval_logs` | test output that decided `target` |

`role="ai"` is the agent's reasoning plus its command in a ``` fence. `role="user"` is the environment's output, not a human.

## Known rough edges

- The error flag in `canary peek` is a naive regex. It fires whenever a file merely contains the word "error". Feature extraction needs a real check against the environment's output format.
- `canary sample` oversamples successes to 35% by default. Re-weight, or draw with `--solved-frac 0`, before quoting any rate.
- `exit_context` is a strong signal (24.8% vs 3.4% solve rate on submitted runs), but it is only known when the run ends, so it can't be a model input.
- Escalating to a bigger model is not testable here. Only 100 issues overlap between 70b and 405b, and 528 between 70b and 8b.

## Layout

```
src/canary/
  cli.py            canary entry point
  data.py           dataset location and local paths
  trajectory.py     parsing a single run
  commands/         one module per subcommand
tests/
docs/archive/       planning notes from when this started as a course project
```
