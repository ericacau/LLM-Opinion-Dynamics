This repository refers to the paper [Language-Driven Opinion Dynamics Model for Agent-Based Simulations](https://arxiv.org/abs/2502.19098) and contains the code of simulations and the results of the case study.
<br/>
The paper proposes a novel framework for performing opinion dynamics simulations with LLM agents. Agents engage in pairwise discussions, each holding an opinion in a scale from strongly disagree to strongly agree, and trying to persuade each other.
Agents can eventually change their opinion, thus enabling the analysis of both opinion trends and arguments expressed by agents in natural language.

## Content
The repository is organized as follows:
+ **annotation**: code for annotating the linguistic data;
+ **data**: folder containing the opinion trends and the texts generated with the simulations
+ **annotated_output**: contains the data annotated with logical fallacies
+ **analyses** folder with the code of the logical fallacies analysis

### Code for simulations with LLM agents is available at the following [link](https://github.com/ericacau/LLM_agents_opinion_dynamics)
