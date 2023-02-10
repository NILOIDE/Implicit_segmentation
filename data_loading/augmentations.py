from abc import abstractmethod
from typing import Tuple, List, Dict

import numpy as np
import random
import math


class AbstractAug:

    @abstractmethod
    def num_parameters(self):
        pass


class TranslateCoords(AbstractAug):
    def __init__(self, x_lim=None, y_lim=None, z_lim=None, t_lim=None):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        self.t_lim = t_lim
        self.num_parameters = 0
        if x_lim is not None:
            assert isinstance(x_lim, float)
            self.num_parameters += 1
        if y_lim is not None:
            assert isinstance(y_lim, float)
            self.num_parameters += 1
        if z_lim is not None:
            assert isinstance(z_lim, float)
            self.num_parameters += 1
        if t_lim is not None:
            assert isinstance(t_lim, float)
            self.num_parameters += 1

    def __call__(self, data: Dict[str, np.ndarray]) -> List[float]:
        coords = data["coords"]
        params = []
        if self.x_lim is not None:
            x = random.uniform(-self.x_lim, self.x_lim)
            coords[..., 0] += x
            params.append(x)
        if self.y_lim is not None:
            y = random.uniform(-self.y_lim, self.y_lim)
            coords[..., 1] += y
            params.append(y)
        if self.z_lim is not None:
            z = random.uniform(-self.z_lim, self.z_lim)
            coords[..., 2] += z
            params.append(z)
        if self.t_lim is not None:
            t = random.uniform(-self.t_lim, self.t_lim)
            coords[..., 3] += t
            params.append(t)
        data["coords"] = coords
        return params


class RotateCoords(AbstractAug):
    def __init__(self, x_lim=None, y_lim=None, z_lim=None):
        """
        Rotate coordinates around 3d space. Input range should be [0,1[
        """
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        self.num_parameters = 0
        if x_lim is not None:
            assert isinstance(x_lim, float)
            self.num_parameters += 1
        if y_lim is not None:
            assert isinstance(y_lim, float)
            self.num_parameters += 1
        if z_lim is not None:
            assert isinstance(z_lim, float)
            self.num_parameters += 1

    def gen_rot_matrix(self, thetas: List[float]) -> np.ndarray:
        radians = np.array(thetas) * 2 * np.pi
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(radians[0]), -math.sin(radians[0])],
                        [0, math.sin(radians[0]), math.cos(radians[0])],
                        ])
        R_y = np.array([[math.cos(radians[1]), 0, math.sin(radians[1])],
                        [0, 1, 0],
                        [-math.sin(radians[1]), 0, math.cos(radians[1])],
                        ])
        R_z = np.array([[math.cos(radians[2]), -math.sin(radians[2]), 0],
                        [math.sin(radians[0]), math.cos(radians[0]), 0],
                        [0, 0, 1]
                        ])
        return np.dot(R_z, np.dot(R_y, R_x))

    def __call__(self, data: Dict[str, np.ndarray]):
        # TODO: Now only works for 3d coords
        coords = data["coords"]
        coords_ = coords.reshape(-1, 3)
        theta_1, theta_2, theta_3 = None, None, None
        return_params = []
        if self.x_lim is not None:
            theta_1 = random.uniform(-self.x_lim, self.x_lim)
            return_params.append(theta_1)
        if self.y_lim is not None:
            theta_2 = random.uniform(-self.y_lim, self.y_lim)
            return_params.append(theta_2)
        if self.z_lim is not None:
            theta_3 = random.uniform(-self.z_lim, self.z_lim)
            return_params.append(theta_3)
        R = self.gen_rot_matrix([theta_1 if theta_1 is not None else 0,
                                 theta_2 if theta_2 is not None else 0,
                                 theta_3 if theta_3 is not None else 0,
                                 ])
        data["coords"] = np.dot(coords_, R).reshape(coords.shape)
        return return_params


class GammaShift(AbstractAug):
    def __init__(self, gamma_lim=(0.7, 1.4)):
        assert isinstance(gamma_lim, (tuple, list))
        self.gamma_lim = gamma_lim
        self.num_parameters = 1

    def __call__(self, data: Dict[str, np.ndarray]) -> List[float]:
        image = data["image"]
        gamma = 1
        if self.gamma_lim is not None:
            gamma = random.uniform(self.gamma_lim[0], self.gamma_lim[1])
            image = np.power(image, gamma)
        data["image"] = image
        return [gamma - 1]  # Center parameter at 0
