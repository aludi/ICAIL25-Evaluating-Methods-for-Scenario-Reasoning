# ICAIL25-Evaluating-Methods-for-Scenario-Reasoning
Evaluating Methods for Scenario Reasoning using Bayesian Networks in Exhaustive and Non-Exhaustive Settings


# Run instructions
pip install -r requirements.txt

python3 main.py

This will run the simulation for 9 different parameter settings, in both exhaustive and non-exhaustive settings.
If you only want to run the simulation, comment out lines 155-160 in main.py and uncomment "run_visual()" in line 160.

It takes quite a long time to run.

You might get a problem with cairo, in order to fix that, I do the following:
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/Cellar/cairo/1.18.4/lib
(but this depends on os)


# Abstract
Tunnel vision and confirmation bias can lead to miscarriages of justice. A way to avoid tunnel vision is to consider your evidence in light of more than one scenario. Alternative scenarios allow us to consider how probable each scenario is, compared to the other considered scenarios. Bayesian Networks have been proposed as a formal method for reasoning about probability of alternative scenarios. Specifically, work has been done on modelling alternative scenarios using Bayesian networks with a constraint node, which ensures that the scenarios are mutually exclusive. However, the performance of these methods in situations where not all possible alternative scenarios are modelled, the non-exhaustive setting, has not been investigated.

Since it is impossible to explicitly cover everything that could possibly have happened in a model, it is important to know how these methods handle non-exhaustiveness. We evaluate four of these methods using an agent-based model that simulates an environment in which a crime could occur. Taking this as the ground truth, %we extract relevant events and scenarios as well as their frequencies. 
we compare the different Bayesian network modelling methods on five different aspects related to the quality of the representation of the ground truth as well as computational performance. We find that some proposed methods result in disparities between the ground truth and the predicted posterior probabilities for the scenarios in a non-exhaustive setting. In an exhaustive setting, the proposed methods perform well. The construction approach that models scenarios in terms of conjunctions of events performs well in both non-exhaustive and exhaustive settings.
