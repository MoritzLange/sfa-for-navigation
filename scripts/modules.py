from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class comparableCNN(BaseFeaturesExtractor):
    """
    CNN similar to the hSFA architecture I use.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 32,
        normalized_image: bool = False,
        n_sfa_intermediate_components = 32,
        sfa_rec_field_sizes = [10, 3],
        sfa_strides = [5, 2]
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "comparableCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, n_sfa_intermediate_components, kernel_size=sfa_rec_field_sizes[0], stride=sfa_strides[0], padding=0),
            nn.ReLU(),
            nn.Conv2d(n_sfa_intermediate_components, n_sfa_intermediate_components, kernel_size=sfa_rec_field_sizes[1], stride=sfa_strides[1], padding=0),
            nn.ReLU(),
            nn.Conv2d(n_sfa_intermediate_components, n_sfa_intermediate_components, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Update the output dimensionality
        self._features_dim = n_flatten

        # I'm not adding these two linear layers because when building the ActorCriticPolicy instance, two layers will be added automatically
        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # return self.linear(self.cnn(observations))
        return self.cnn(observations)