# ACMR_for_Humanoid
This repository contains the implementation of the Actor-Critic Model Reuse (ACMR) framework, a novel approach to transfer learning in Deep Reinforcement Learning (DRL) that addresses adaptation challenges when environments undergo significant changes in state-action space dimensions, such as sensor and actuator reduction.

# Overview

In DRL, adapting an agent to environmental changes that alter its state-action space can be challenging, often requiring extensive retraining. ACMR introduces a transfer learning methodology that reuses pre-trained Actor-Critic models to accelerate adaptation in target environments with modified state-action spaces. ACMR's configurations—hidden layer reuse, layer freezing, and network layer expansion—enable effective transfer and adaptation across environments.

# Key Features
1. Model Transfer Configurations: Supports multiple ACMR configurations:
   - Hidden Layer Transfer: Retains learned feature representations in hidden layers.
   - Layer Freezing: Fixes specific pre-trained layers to preserve knowledge and reduce training time.
   - Layer Expansion: Adds flexible layers for input-output adaptation in environments with changed dimensions.
2. Soft Actor-Critic (SAC) Integration: Built on the SAC algorithm.
3. Benchmarking with Gymnasium's Humanoid Environment: Tests include simulated state-action reductions (e.g., "ArmlessHumanoid" configuration), providing benchmarks for ACMR's efficiency in adaptation.

# Credits

This code is based on the [CleanRL repository](https://github.com/vwxyzjn/cleanrl), specifically [this script](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) by [vwxyzjn](https://github.com/vwxyzjn). CleanRL is licensed under the MIT License.

## Prerequisites

- Python > 3.10

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/feline-malin/ACMR_for_Humanoid.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ACMR_for_Humanoid
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ``` 

## Usage

### Start experiments

1. Configure environment in MalinHumanoidEnv.py
2. Configure Scenario in sac_continous_action_transfer.py
3. Configure save paths in sac_continous_action_transfer.py
4. Start experiment by running sac_continous_action_transfer.py

To only save a model, use the sac_continous_action_save_model.py

## Author
Feline Malin Barg