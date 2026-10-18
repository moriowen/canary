# Project direction: CS 7641

Reference note. The proposal drafts keep changing, so this records what the project is for and what has to stay true across any rewrite. Written Aug 26 2026, before the team forms on Sept 18.

Companion files: [[Sample project proposal]] for the current draft, [[Award-winning ML projects]] for what strong work looks like in this course, and [[CS 7641 - Machine Learning]] for mechanics and deadlines.

---

## 1. The domain constraint

The project sits in AI and LLM systems. Not biology, not medicine, not sports, and nothing outside computing.

This is deliberate, and it is the most consequential decision in the project. Every domain charges a tax: the weeks spent learning enough of a field to tell whether a model's output makes sense before any real work can start. On a six or seven week clock that tax is fatal. The award-winning medical projects in [[Award-winning ML projects]] were built by teams who already had the domain knowledge and were not paying it during the semester.

Time is only part of it. You cannot evaluate a model in a field where you cannot smell a wrong answer, and that instinct is what separates a good project from a mediocre one. It also happens to be the one thing that cannot be acquired in six weeks.

### Domains that cost me nothing

LLM and agent behaviour is the current pick: traces, failures, cost, tool calls. Software engineering artifacts are the closest alternative, since PRs, CI logs, issue trackers, build failures, and commit history are things I have read professionally for three years. Video understanding needs almost no ramp-up given two published papers ([[Research]]). Distributed and transaction systems are free on the domain side from the JPMC settlement work ([[JPMC Book of Work]]), though public data is the constraint there.

### Out of scope

Anything that requires learning a non-CS field before I can judge whether an output is plausible: bio, medicine, chemistry, climate, clinical work of any kind. Also out are projects whose entire interest is the domain rather than the method.

### The interest, stated plainly

I want to work in the LLM and AI space after this degree. The project should be something I can talk about in an interview, or to a professor, as evidence that I have worked seriously on AI systems rather than completed a course exercise. That rules out generic Kaggle-style modelling however cleanly it would score.

---

## 2. The problem

In one line: AI coding agents fail about five out of six times and burn a full run's compute doing it. Using 80,036 recorded attempts, discover how they fail and predict failure early enough to stop a doomed run.

### The setup

An AI coding agent gets a real GitHub bug and works in a loop, thinking, running a command, reading the output, thinking again, for a median of 31 messages. At the end it submits a patch and a test suite decides whether the fix worked. Someone recorded 80,036 of these attempts and published them.

### The two halves

The supervised half predicts. From the first *k* turns of a transcript only, predict the final outcome. Each transcript collapses to one row of numbers: error count, repeated commands, output volume, command types. The label is already in the data. What comes out is a probability rather than a verdict, and the abort decision is made by choosing a threshold on it. Picking that threshold trades compute saved against successful runs destroyed, which is why the sweep across thresholds is a deliverable.

The unsupervised half discovers. The dataset ships an `exit_status` field that explains mechanical failures: ran out of context, hit a cost cap, crashed. But 59,880 runs (74.8%) finish normally, submit a patch, and are simply wrong. For every one of those, `exit_status` says nothing, and no taxonomy of those failures exists. Clustering builds one.

Clustering returns no decision. Its raw output is a cluster number per run, which means nothing until someone reads about 20 transcripts per cluster and names it: never ran the tests, edited the wrong file, looped on one command. That naming step is the finding, and no algorithm does it for us. The named groups then serve two further purposes: they become an input column to the predictor, and their per-cluster abort rates show whether we are systematically killing one kind of run.

The two halves need each other. Because a new run can be assigned to its nearest cluster from its first *k* turns, the discovered failure types become an input to the prediction model, so instead of asking only whether a run will fail, it can learn that a run looks like the never-ran-the-tests failure mode. Delete the clustering and the project gets worse. Most CS 7641 projects fail that test, because the rubric forces unsupervised learning in and teams bolt on a scatter plot.

The combined output is what makes the report worth reading. The predictor alone says a run will probably fail. With clustering it says a run will probably fail and is failing by never running the tests. The first supports an abort; the second says what to fix.

### The complication that defines the project

Of 3,591 tasks, 2,753 (76.7%) are never solved in any attempt. A model could therefore score well by recognising that a task is hopeless rather than that a run is going badly, which is useless for an abort decision: it would fire at step 0 and never change its mind.

The fix comes from the same data. There are 807 tasks (22.5%) with both outcomes, covering 34,922 runs. Same bug, sometimes fixed, sometimes not. Evaluating there holds difficulty constant.

Reporting both numbers, full-dataset and within-task, is the core of the project. The gap between them is the confound, measured.

---

## 3. Locked and flexible

Changing any of these means restarting, so they are locked. The domain is AI and LLM systems. Data must be public and available on day one, with no collection phase, no annotation project, and no API spend. Both halves must carry weight, and the unsupervised half must survive the deletion test. Evaluation discipline holds: grouped splits, an explicit dumb baseline, and never accuracy on imbalanced data. There has to be a decision at the end rather than only a metric.

These are expected to move, and two of them already have. The exact wording of the problem statement. Which failure taxonomy emerges and how many clusters it has. Values of *k*, choice of models, and whether the transfer tests survive scope cuts. Whether `nebius/SWE-agent-trajectories` stays the primary dataset, which section 5 covers.

The rule on timing: lock the dataset before the proposal on Oct 9, and keep the problem statement moving until the midterm on Nov 13. Changing datasets after Oct 9 costs weeks that do not exist. Sharpening the question costs nothing, and the midterm deliverable is defined as extending the proposal.

---

## 4. Test for any candidate problem statement

Any replacement has to pass all five.

1. The decision sentence. Can I write "someone must decide X, and today they decide it by Y"? If not, it is a dataset and not a project.
2. The deletion test. If I removed the unsupervised half, would anything downstream break? "We would lose a figure" is a fail.
3. Data on day one. Public, free, already labeled, no collection phase.
4. The plausibility test. Can I look at a model output and tell whether it makes sense? If not, wrong domain.
5. The dumb baseline. Is there an obvious non-ML rule, and have I named it myself? A mentor naming it first makes the project look naive.

The current problem statement passes all five. Against the six traits shared by the winners in [[Award-winning ML projects]] it has a real decision, unsupervised learning that carries weight, non-Kaggle data, a domain-specific metric in compute saved at fixed success rate, and a quantified self-damaging limitation in the confound gap. The trait it cannot claim yet is a documented mid-project pivot, which gets written when it happens.

---

## 5. Fallbacks in the same domain

If the current problem statement dies, these reuse the same data, skills, and constraints. They are ordered by how little would need rebuilding.

1. Pure discovery. Drop prediction and build the failure taxonomy alone, at depth. Weaker, since there is no decision at the end, but close to guaranteed to produce a result.
2. Step-level cost routing. For each step, can a cheap model handle it or is the expensive one needed? Same trace data, different question, easier to scope.
3. Cross-scaffold generalisation. Do failure modes transfer between agent frameworks? Uses SWE-smith as a second scaffold. More of a study than a system.
4. Tool-call failure prediction. Predict whether a command will error before running it. Narrowest and safest, and the least interesting to write up.
5. Software engineering artifacts, if agent data fails entirely: PR review prioritisation, flaky-test detection, or build-failure triage. Different data, same zero domain tax.

Rejected: escalation to a larger model. Measured task overlap between backbone scales is only 100 tasks, too thin to support any claim.

---

## 6. Constraints that bind any version

The team forms Sept 18 and work starts around Sept 29. The proposal is due Oct 9 and the midterm Nov 13, but presentations begin the week of Nov 16, and that is the deadline that binds rather than the Dec 8 report date. A3 on Nov 20, A4 on Dec 8, and Thanksgiving recess from Nov 23 to 27 all compete for the same time. That leaves roughly six to seven usable weeks.

Five people means the work has to split into five parallel lanes, or three of them become passengers and CATME will surface it.

The rubric requires supervised and unsupervised learning, three or more preprocessing methods, three or more models, three or more metrics, and three or more references. Project deliverables have no late window.

I am not chasing the award. The award traits are being used as a quality bar rather than a goal. The Award Eligibility Form is opt-in, so opt out on purpose rather than by forgetting.

---

## 7. Current state

The proposal is at v3 with sections 1 to 4 at 783 of 800 words: [[Sample project proposal]].

The local repo is `~/dev/Canary`, with a venv, three working scripts, a 6,000-row sample pulled, and every headline number verified against the real data rather than estimated.

Next is `04_features.py`: the feature table at k = 5, the grouped split, and the two baselines.

Open question: whether to validate the dataset choice against alternatives while switching is still free, before Oct 9, or commit now.



Hey everyone!

I’m Atharva, a first-year MSCS (ML) student with undergraduate Computer Vision research experience and three years as a software engineer building financial systems at JPMorgan.

I’m interested in AI systems and LLM inference. I have an initial project idea around optimizing local LLM inference cost.

I’m looking to form a group of five or join a group already working on something similar. If you’re interested to team up you can reach out at: amohite8@gatech.edu