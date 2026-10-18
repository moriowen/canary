# Problem statement: Can we spot failing self-hosted coding agents early?

An AI coding agent receives a real GitHub bug and tries to fix it. It works in a loop: it thinks, runs a command, reads the output, and tries again. After roughly 31 messages, it submits a patch. The project's tests decide whether the patch works.

The problem is that failed runs often use the same full budget as successful runs. For self-hosted models, that means wasted GPU time, power, and queue time. If we can tell early that a run is unlikely to succeed, we can stop it and free that capacity.

## Dataset

We will use the public [SWE-agent-trajectories dataset](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), which contains 80,036 agent attempts across 3,591 real bugs. These are open-weight models run on local hardware, not paid coding services.

| | |
| --- | --- |
| Attempts | 80,036 across 3,591 bugs |
| Tries per bug | 22 on average, up to 125 |
| Successful attempts | 13,389 (16.7%) |
| Average length | About 31 messages |
| Quiet failures | 59,880 (74.8%) |
| Model | Open-weight Llama throughout; 93% use one size |

Most failed attempts end normally: the agent submits a patch, but the patch does not pass the tests. The dataset records the sequence of actions and outputs, so it lets us ask whether the first few turns show that a run is heading toward failure.

## Questions

1. Can early turns predict that an agent run is likely to fail?
2. Can we identify common failure patterns, such as never running tests, editing the wrong file, repeating a command, or fixing a symptom instead of the cause?
3. How much compute time could an early warning save without stopping successful runs?

## Why this is a good ML problem

The problem has a real decision at the end: whether to stop a running agent. Right now, a person must either watch the run or set a fixed time limit. A model can instead return a probability of failure from the first few turns. Choosing a threshold on that probability makes the trade-off clear: a lower threshold saves more compute but may stop runs that would have succeeded.

It also has a useful baseline. Rules such as "stop after three consecutive errors" or "stop after 15 steps" require no machine learning. Any learned approach should show that it improves on those rules.

The data is public, already labeled by project tests, and ready to use. That lets us spend time on the analysis rather than collecting traces or manually labeling success and failure.

## Why supervised and unsupervised learning both matter

**Supervised learning answers whether a run will fail.** The dataset's `target` field tells us whether the agent fixed the bug. From the first *k* turns, we can build features such as error count, repeated commands, output volume, and command types, then train a classifier to estimate the chance of failure.

**Unsupervised learning answers how it is failing.** The `exit_status` field identifies mechanical endings such as context exhaustion or a cost cap. It does not explain the 59,880 runs that submit a patch normally but still fail the tests. Those runs have no failure-type labels. Clustering their early traces can reveal patterns that we can inspect and name, such as never running tests or repeatedly using the same command.

The two parts are connected rather than separate course requirements. After clustering discovers the failure patterns, a new prefix can be assigned to the nearest cluster. That cluster assignment becomes another input to the supervised model. The classifier can then learn more than error counts: it can learn that a run resembles a particular failure pattern and rarely recovers.

The clusters also check the classifier. Reporting abort and success rates for each cluster can reveal whether the model is systematically stopping a type of run that sometimes succeeds. The supervised model supports the stop-or-continue decision; the clusters explain the behavior behind it and point to what should be improved.

## Important limitation

Many bugs in the dataset are never fixed: 2,753 of 3,591 were unsuccessful in every attempt. A model could look good simply by learning which bugs are hard. That would not tell us whether a particular agent run is going badly.

To separate bug difficulty from agent behavior, we can focus on the 807 bugs with both successful and failed attempts, covering nearly 35,000 runs. For those bugs, the task stays the same while the outcome changes.

The dataset is also dominated by one model size. We can describe how agents in that setup fail, but we cannot make reliable claims about how failure changes across model sizes.
