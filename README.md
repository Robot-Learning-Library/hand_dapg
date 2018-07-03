# DAPG for Dexterous Hand Manipulation

This repository is to accompany the paper [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://arxiv.org/abs/1709.10087), presented at RSS 2018.

## Organization

The overall project is organized into three repositories:

1. [mjrl](https://github.com/aravindr93/mjrl) provides a suite of learning algorithms for various continuous control tasks simulated in MuJoCo. This includes the NPG implementation and the DAPG algorithm used in the paper.
2. [mj_envs](https://github.com/vikashplus/mj_envs) provides a suite of continuous control tasks simulated in MuJoCo, including the dexterous hand manipulation tasks used in the paper.
3. [handrl_rss2018](https://github.com/aravindr93/dapg) (this repository) serves as the landing page and contains the human demonstrations and pre-trained policies for the tasks.

This modular organization was chosen to allow for rapid and independent developments along different directions such as algorithms and interesting tasks, and also to facilitate sharing of results with the broader research community.

## Getting started

Each repository above contains detailed setup instructions. 
1. *Step 1:* Install [mjrl](https://github.com/aravindr93/mjrl), using instructions in the repository ([direct link](https://github.com/aravindr93/mjrl/tree/master/setup)). `mjrl` comes with an anaconda environment which helps to easily import and use a variety of mujoco tasks.
2. *Step 2:* Clone [mj_envs](https://github.com/vikashplus/mj_envs) and follow instructions in the repository for setting various paths correctly. Once `mjrl` is setup, installing `mj_envs` should be straightforward.
3. *Step 3:* After setting up `mjrl` and `mj_envs`, clone this repository and use the following commands to visualize the demonstrations and pre-trained policies.

```
$ cd dapg
$ python utils/visualize_demos --env_name relocate-v0
$ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle
```

*NOTE:* If the visualization results in a GLFW error, this is because `mujoco-py` does not see some graphics drivers correctly. This can usually be fixed by explicitly loading the correct drivers before running the python script. See [this page](https://github.com/aravindr93/mjrl/tree/master/setup#known-issues) for details.