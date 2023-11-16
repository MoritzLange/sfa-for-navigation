from tqdm import tqdm
import pickle
import os

import numpy as np

from sksfa import HSFA

import gymnasium as gym
import miniworld

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TransformObservation

from wandb.integration.sb3 import WandbCallback
import wandb

from modules import comparableCNN
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', dest='env', type=str, default="MiniWorld-StarMazeArm-v0") # "MiniWorld-StarMazeArm-v0", "MiniWorld-StarMazeRandom-v0", "MiniWorld-WallGap-v0", "MiniWorld-FourColoredRooms-v0"
parser.add_argument('-m', '--repl_mode', dest='repl_mode', type=str, default="sfa") # sfa, pca, cnn, cnn_comp (cnn is sb3 default, cnn_comp has same structure as sfa extractor)
args = parser.parse_args()

config = {
    "env_name": args.env, 
    "repl_mode": args.repl_mode,
    "n_steps": 128,
    "gamma": 0.99,
    "learning_rate": 2.5e-4,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.1,
    "batch_size": 128,
    "policy_kwargs": None, # dict(net_arch=[64, 64])
}

# Decide length of training
if config["env_name"] in ["MiniWorld-StarMazeRandom-v0", "MiniWorld-WallGap-v0", "MiniWorld-FourColoredRooms-v0"]:
    config["total_timesteps"] = 2e6
else:
    config["total_timesteps"] = 1e6

# Decide features extractor
if config["repl_mode"] in ["sfa", "pca"]:
    config["policy_type"] = "MlpPolicy"
else:
    config["policy_type"] = "CnnPolicy"
    if config["repl_mode"] == "cnn_comp":
        config["policy_kwargs"] = dict(features_extractor_class = comparableCNN)

os.makedirs("./.wandb", exist_ok=True)
run = wandb.init(
    entity="xxxx", # Use the name of your wandb entity here
    project="sfa_project",
    config=config,
    dir = "./.wandb",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # mode = "disabled"
    # save_code=True,  # optional
)


# Create the environments
env = gym.make(config["env_name"], view="agent", render_mode="rgb_array")

if config["env_name"] in ["MiniWorld-StarMazeRandom-v0", "MiniWorld-StarMazeArm-v0"]:
    suffix = "starmaze"
elif config["env_name"] in ["MiniWorld-WallGap-v0"]:
    suffix = "wallgap"
elif config["env_name"] in ["MiniWorld-FourColoredRooms-v0"]:
    suffix = "fourcoloredrooms"

match config["repl_mode"]:
    case "sfa":
        with open(f'../transformers/trained_hsfa_transformer_{suffix}.pickle', 'rb') as handle:
            hsfa = pickle.load(handle)
        hsfa.verbose = 0
        env = Monitor(TransformObservation(env, lambda obs: hsfa.transform(obs[None])))
        env.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,32))
    case "pca":
        with open(f'../transformers/trained_pca_transformer_{suffix}.pickle', 'rb') as handle:
            pca = pickle.load(handle)
        env = Monitor(TransformObservation(env, lambda obs: pca.transform(obs.flatten()[None])))
        env.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,32))
    case "cnn":
        env = Monitor(env)


# Train the RL algorithm
model = PPO(config["policy_type"],
            env,
            verbose=1,
            tensorboard_log=f".logs/.runs/{run.id}",
            n_steps=config["n_steps"],
            gamma=config["gamma"],
            learning_rate=config["learning_rate"],
            gae_lambda=config["gae_lambda"],
            ent_coef=config["ent_coef"],
            clip_range=config["clip_range"],
            batch_size=config["batch_size"],
            policy_kwargs=config["policy_kwargs"]
            )

model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(gradient_save_freq=10000,
                                   model_save_path=f".logs/models/{run.id}",
                                   log="all",
                                   verbose=2),
            progress_bar=True)

run.finish()
