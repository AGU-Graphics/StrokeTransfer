# MIT License
#
# Copyright (c) 2022  Hideki Todo, Kunihiko Kobayashi, Jin Katsuragi, Haruna Shimotahira, Shizuo Kaji, Yonghao Yue
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from glumpy import glm

from util.dict_data import loadDict


class LocationInfo:
    def __init__(self, data):
        self.data = data
        self.location = np.array(data["location"])
        y = self.location[2]
        z = -self.location[1]
        self.location[1] = y
        self.location[2] = z


class TransformInfo(LocationInfo):
    def __init__(self, data):
        super().__init__(data)
        self.world = np.array(data["world"])


class CameraInfo(TransformInfo):
    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.project_mat = np.array(data["project_mat"])
        self.flocal_length = data["flocal_length"]
        self.near = data["near"]
        self.far = data["far"]
        self.angle_x = data["angle_x"]
        self.film_x = data["film_x"]
        self.film_y = data["film_y"]


class BlenderInfo:
    def __init__(self, file_path):
        self.data = loadDict(file_path)
        self.camera = CameraInfo(self.data["camera"])

    def MVPMatrix(self):
        model_mat = np.eye(4, dtype=np.float32)
        model_mat = glm.rotate(model_mat, 90, 1, 0, 0)

        view_mat = self.camera.world
        view_mat = np.linalg.inv(view_mat)
        view_mat = view_mat.T

        project_mat = self.camera.project_mat
        project_mat = project_mat.T

        return model_mat, view_mat, project_mat
