# This script is based on the SAC implementation in the CleanRL repository:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
# Modifications have been made to fit the specific requirements of this work.

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

"""
SAC from cleanrl - changed by Malin Barg 11.03.2024
--> Script for starting simulation and load and save models for transfer learning

Experiment configuration:
- args.env_id
- args.exp_name 
- paths
- freeze
- trans_layers
"""

# Register the environment
gym.envs.registration.register(
    id='MalinHumanoid-v0',
    entry_point='envs.MalinHumanoidEnv:MalinHumanoidEnv',
    max_episode_steps=1000  # limits for the TimeLimit-Wrapper
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def custom_episode_trigger(episode_id: int) -> bool:
    """Custom episode trigger function for video."""
    trigger_episodes = [500, 1000, 1200, 1400, 1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590, 1600, 1602,
                        1605, 1607, 1610, 1612, 1615, 1620, 1625, 1630]
    return episode_id in trigger_episodes


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, source_obs_space, source_action_space, freeze_fc2=False, freeze_fc_layers=False,
                 freeze_adaption_layer=False):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        if freeze_adaption_layer:
            # Freeze the adaption layer
            for param in self.adaption_layer.parameters():
                param.requires_grad = False

        if freeze_fc_layers:
            # freeze all hidden fc layers
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.fc2.parameters():
                param.requires_grad = False
            for param in self.fc3.parameters():
                param.requires_grad = False

        if freeze_fc2:
            for param in self.fc2.parameters():
                param.requires_grad = False

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AdaptionSoftQNetwork(nn.Module):
    def __init__(self, env, source_obs_space, source_action_space, freeze_fc2=False, freeze_fc_layers=False,
                 freeze_adaption_layer=False):
        super().__init__()
        self.adaption_layer = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            np.array(source_obs_space.shape).prod() + np.prod(source_action_space.shape))
        self.fc1 = nn.Linear(np.array(source_obs_space.shape).prod() + np.prod(source_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        if freeze_adaption_layer:
            # Freeze the adaption layer
            for param in self.adaption_layer.parameters():
                param.requires_grad = False

        if freeze_fc_layers:
            # freeze all hidden fc layers
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.fc2.parameters():
                param.requires_grad = False
            for param in self.fc3.parameters():
                param.requires_grad = False

        if freeze_fc2:
            for param in self.fc2.parameters():
                param.requires_grad = False

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.adaption_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# The SoftQNetwork class represents a Q-function neural network in the SAC algorithm.
# It takes the state and action as inputs and outputs the corresponding Q-value.
# This network is used to estimate the expected cumulative reward for taking a specific action in a given state.

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, source_obs_space, source_action_space, freeze_actor_fc2=False, freeze_actor_all=False):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

        if freeze_actor_fc2:
            for param in self.fc2.parameters():
                param.requires_grad = False

        if freeze_actor_all:
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.fc2.parameters():
                param.requires_grad = False
            for param in self.fc_mean.parameters():
                param.requires_grad = False
            for param in self.fc_logstd.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# the Actor class defines the policy network in the SAC algorithm.
# It takes a state as input and outputs the mean and log standard deviation of a Gaussian policy distribution.
# The get_action method then samples an action from this distribution,
# applies necessary transformations, and returns the action, log probability, and rescaled mean.

class AdaptionActor(nn.Module):
    def __init__(self, env, source_obs_space, source_action_space, freeze_actor_fc2=False, freeze_actor_all=False):
        super().__init__()
        self.adaption_layer_1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), np.array(source_obs_space.shape).prod())
        self.fc1 = nn.Linear(np.array(source_obs_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(source_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(source_action_space.shape))
        self.adaption_layer_mean = nn.Linear(np.prod(source_action_space.shape), np.prod(env.single_action_space.shape))
        self.adaption_layer_logstd = nn.Linear(np.prod(source_action_space.shape), np.prod(env.single_action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

        if freeze_actor_fc2:
            for param in self.fc2.parameters():
                param.requires_grad = False

        if freeze_actor_all:
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.fc2.parameters():
                param.requires_grad = False
            for param in self.fc_mean.parameters():
                param.requires_grad = False
            for param in self.fc_logstd.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.adaption_layer_1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        mean = self.adaption_layer_mean(mean)
        log_std = self.fc_logstd(x)
        log_std = self.adaption_layer_logstd(log_std)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

def save_model(actor, qf1, qf2, save_path, source_obs_space, source_action_space):
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'qf1_state_dict': qf1.state_dict(),
        'qf2_state_dict': qf2.state_dict(),
        'source_obs_space': source_obs_space,
        'source_action_space': source_action_space,
    }, save_path)


def load_model(actor, qf1, qf2, load_path, trans_layers):
    checkpoint = torch.load(load_path)

    if trans_layers['actor_all']:
        actor_state_dict = checkpoint['actor_state_dict']
        filtered_actor_state_dict = {k: v for k, v in actor_state_dict.items() if
                                     'fc1' in k or 'fc2' in k or 'fc_mean' in k or 'fc_logstd' in k}
        actor.load_state_dict(filtered_actor_state_dict, strict=False)

    if trans_layers['q_all']:
        qf1_state_dict = checkpoint['qf1_state_dict']
        filtered_qf1_state_dict = {k: v for k, v in qf1_state_dict.items() if 'fc1' in k or 'fc2' in k or 'fc3' in k}
        qf1.load_state_dict(filtered_qf1_state_dict, strict=False)

        qf2_state_dict = checkpoint['qf2_state_dict']
        filtered_qf2_state_dict = {k: v for k, v in qf2_state_dict.items() if 'fc1' in k or 'fc2' in k or 'fc3' in k}
        qf2.load_state_dict(filtered_qf2_state_dict, strict=False)

    if trans_layers['actor_fc1']:
        actor_state_dict = checkpoint['actor_state_dict']
        filtered_actor_state_dict = {k: v for k, v in actor_state_dict.items() if 'fc1' in k}
        actor.load_state_dict(filtered_actor_state_dict, strict=False)

    if trans_layers['actor_fc2']:
        actor_state_dict = checkpoint['actor_state_dict']
        filtered_actor_state_dict = {k: v for k, v in actor_state_dict.items() if 'fc2' in k}
        actor.load_state_dict(filtered_actor_state_dict, strict=False)

    if trans_layers['q_fc1']:
        qf1_state_dict = checkpoint['qf1_state_dict']
        filtered_qf1_state_dict = {k: v for k, v in qf1_state_dict.items() if 'fc1' in k}
        qf1.load_state_dict(filtered_qf1_state_dict, strict=False)

        qf2_state_dict = checkpoint['qf2_state_dict']
        filtered_qf2_state_dict = {k: v for k, v in qf2_state_dict.items() if 'fc1' in k}
        qf2.load_state_dict(filtered_qf2_state_dict, strict=False)

    if trans_layers['q_fc2']:
        qf1_state_dict = checkpoint['qf1_state_dict']
        filtered_qf1_state_dict = {k: v for k, v in qf1_state_dict.items() if 'fc2' in k}
        qf1.load_state_dict(filtered_qf1_state_dict, strict=False)

        qf2_state_dict = checkpoint['qf2_state_dict']
        filtered_qf2_state_dict = {k: v for k, v in qf2_state_dict.items() if 'fc2' in k}
        qf2.load_state_dict(filtered_qf2_state_dict, strict=False)

    if trans_layers['q_fc3']:
        qf1_state_dict = checkpoint['qf1_state_dict']
        filtered_qf1_state_dict = {k: v for k, v in qf1_state_dict.items() if 'fc3' in k}
        qf1.load_state_dict(filtered_qf1_state_dict, strict=False)

        qf2_state_dict = checkpoint['qf2_state_dict']
        filtered_qf2_state_dict = {k: v for k, v in qf2_state_dict.items() if 'fc3' in k}
        qf2.load_state_dict(filtered_qf2_state_dict, strict=False)


def read_source_information(load_path):
    checkpoint = torch.load(load_path)

    # Extract information from the loaded model
    source_obs_space = checkpoint.get("source_obs_space", None)
    source_action_space = checkpoint.get("source_action_space", None)

    # Print or return the information
    print("Source Environment Observation Space:", source_obs_space)
    print("Source Environment Action Space:", source_action_space)

    return source_obs_space, source_action_space


def run_experiment(args, run_name, paths, freeze, trans_layers, adaption):
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # read information about source model
    if paths['load_path'] is not None:
        if adaption['Q-network'] is True:
            source_obs_space, source_action_space = read_source_information(paths['load_path'])
        elif adaption['Actor-network'] is True:
            source_obs_space, source_action_space = read_source_information(paths['load_path'])
        else:
            source_obs_space = None
            source_action_space = None

    if adaption['Q-network'] is True:
        qf1 = AdaptionSoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'],
                                   freeze['freeze_q_all'],
                                   freeze['freeze_q_adaption']).to(device)
        qf2 = AdaptionSoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'],
                                   freeze['freeze_q_all'],
                                   freeze['freeze_q_adaption']).to(device)
        qf1_target = AdaptionSoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'],
                                          freeze['freeze_q_all'], freeze['freeze_q_adaption']).to(device)
        qf2_target = AdaptionSoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'],
                                          freeze['freeze_q_all'], freeze['freeze_q_adaption']).to(device)
    else:
        qf1 = SoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'], freeze['freeze_q_all'],
                           freeze['freeze_q_adaption']).to(device)
        qf2 = SoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'], freeze['freeze_q_all'],
                           freeze['freeze_q_adaption']).to(device)
        qf1_target = SoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'],
                                  freeze['freeze_q_all'], freeze['freeze_q_adaption']).to(device)
        qf2_target = SoftQNetwork(envs, source_obs_space, source_action_space, freeze['freeze_q_fc2'],
                                  freeze['freeze_q_all'], freeze['freeze_q_adaption']).to(device)

    if adaption['Actor-network'] is True:
        actor = AdaptionActor(envs, source_obs_space, source_action_space, freeze['freeze_actor_fc2'],
                      freeze['freeze_actor_all']).to(device)
    else:
        actor = Actor(envs, source_obs_space, source_action_space, freeze['freeze_actor_fc2'],
                      freeze['freeze_actor_all']).to(device)

    if paths['load_path'] is not None:
        load_model(actor, qf1, qf2, paths['load_path'], trans_layers)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Create a DataFrame to store episode information
    columns = []
    infos_df = pd.DataFrame(columns=columns)
    episode_df = pd.DataFrame(columns=columns)
    # observation_df = pd.DataFrame(columns=columns3)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                # save episode information in csv
                new_eps = pd.DataFrame(
                    {"global_step": global_step, "episodic_return": info['episode']['r'], "reward": rewards})
                episode_df = pd.concat([episode_df, new_eps], ignore_index=True)
                episode_df.to_csv(paths['episode_save_path'])

                # save infos in csv
                new_infos = pd.DataFrame(info)
                infos_df = pd.concat([infos_df, new_infos], ignore_index=True)
                infos_df.to_csv(paths['infos_save_path'])

                # new_obs = pd.DataFrame(next_obs)
                # observation_df = pd.concat([observation_df, new_obs], ignore_index=True)
                # observation_df.to_csv("save_data/observation_df.csv")
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if paths['save_path'] is not None:
        # Save the model with environment information
        save_model(actor, qf1, qf2, paths['save_path'], envs.single_observation_space, envs.single_action_space)

    envs.close()
    writer.close()


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)

    # Experiment HB
    args.env_id = "MalinHumanoid-v0"
    args.exp_name = "test"  # Change experiment name
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # configure paths
    paths = {'load_path': "",
             'save_path': "saved_models/test.pth",
             'episode_save_path': "save_data/test_eps.csv",
             'infos_save_path': "save_data/test_infos.csv"}

    # configure freezing
    freeze = {'freeze_actor_fc2': False,
              'freeze_actor_all': False,
              'freeze_q_fc2': False,
              'freeze_q_all': False,
              'freeze_q_adaption': False}

    # configure transfered layers
    trans_layers = {'actor_all': False,
                    'actor_fc2': False,
                    'actor_fc1': False,
                    'q_all': False,
                    'q_fc2': False,
                    'q_fc1': False,
                    'q_fc3': False}

    # configure adaption layers
    adaption = {'Q-network': False,
                'Actor-network': False}

    run_experiment(args, run_name, paths, freeze, trans_layers, adaption)

