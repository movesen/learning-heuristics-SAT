**Deep Reinforcement Learning for SAT solver heuristics**
----
This repository provides three different implementations of Reinforcement Learning techniques for learning SAT (Boolean Satisfybility) solver heuristics.

Included in the repository:
  1. [REINFORCE w/ Baseline algorithm](https://github.com/movesen/learning-heuristics-SAT/tree/main/SAT_REINFORCE_Baseline)
  2. [Experience Replay implementation](https://github.com/movesen/learning-heuristics-SAT/tree/main/SAT_Experience_Replay)
  3. [REINFORCE algorithm](https://github.com/movesen/learning-heuristics-SAT/tree/main/SAT_REINFORCE), originally implemented by [@yanneta](https://github.com/yanneta)

**Introduction**
----
Local search algorithms are recognized for their effectiveness in addressing large, hard instances of the satisfiability problem (SAT). The success of these algorithms significantly hinges on the application of heuristics to adjust noise parameters and evaluate variables. The best configuration of these heuristics changes across various instance distributions. In this study, we introduce multiple new reinforcement learning methods to develop efficient variable evaluation functions.

**Requirements**
----
- CNFgen
- minisat

  
Install:

```
#CNFgen
pip install CNFgen

#minisat
https://howtoinstall.co/en/minisat
```
