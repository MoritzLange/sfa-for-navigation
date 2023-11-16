from tqdm import tqdm
import pickle

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
import time


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', dest='env', type=str, default="MiniWorld-StarMazeArm-v0") # "MiniWorld-StarMazeArm-v0", "MiniWorld-StarMazeRandom-v0", "MiniWorld-WallGap-v0", "MiniWorld-FourColoredRooms-v0"
parser.add_argument('-m', '--repl_mode', dest='repl_mode', type=str, default="sfa") # sfa, pca, cnn, cnn_comp (cnn is sb3 default, cnn_comp has same structure as sfa extractor)
parser.add_argument('-r', '--run_id', dest='run_id', type=str) # Use a run id from Weights and Biases here
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

# Decide features extractor
if config["repl_mode"] in ["sfa", "pca"]:
    config["policy_type"] = "MlpPolicy"
else:
    config["policy_type"] = "CnnPolicy"
    if config["repl_mode"] == "cnn_comp":
        config["policy_kwargs"] = dict(features_extractor_class = comparableCNN)


# Create the environments
env = gym.make(config["env_name"], view="top", render_mode="human")

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


# Load the RL algorithm
model = PPO.load(f".logs/models/{args.run_id}/model")

obs, _ = env.reset()
counter = 0

max_episode_length = 300
sleep_time_step = 0.03
sleep_time_reset = 1
while True:
    action, _states = model.predict(obs)
    obs, reward, term, trunc, info = env.step(action)
    env.render()
    if term or trunc or (counter>=max_episode_length):
        env.reset()
        counter=0
    if term:
        time.sleep(sleep_time_reset)
    time.sleep(sleep_time_step)
    counter += 1
