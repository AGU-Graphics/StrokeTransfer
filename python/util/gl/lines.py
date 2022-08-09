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
from glumpy import gl, glm, gloo

import util.gl.shaders.mesh as mesh_shader


class GLLines:
    def __init__(self, P0, P1, W=2.0, C=None, shader=mesh_shader):
        self.P0 = P0
        self.P1 = P1
        self.W = W
        self.C = C

        FV = np.hstack((P0, P1))
        FV = FV.reshape(-1, 3)

        lines = gloo.Program(shader.vertex, shader.fragment, count=FV.shape[0])
        self.lines = lines

        lines['position'] = FV

        if C is not None:
            self.setColor(C)

        lines['u_model'] = np.eye(4, dtype=np.float32)
        lines['u_view'] = glm.translation(0, 0, 0.5)
        projection = np.eye(4, dtype=np.float32)
        lines['u_projection'] = projection
        lines['u_normal_mat'] = np.eye(3, dtype=np.float32)

    def setPosition(self, P0, P1):
        self.P0 = P0
        self.P1 = P1

        FV = np.hstack((P0, P1))
        FV = FV.reshape(-1, 3)
        self.FV = FV

        self.lines['position'] = FV

    def setP0(self, P0):
        self.P0 = P0

    def setVector(self, V, scale=1.0):
        P0 = self.P0
        P1 = P0 + scale * V
        self.setPosition(P0, P1)

    def setLineColors(self, C):
        FC = np.array([[Ci, Ci] for Ci in C])
        FC = FC.reshape(-1, 4)
        self.lines['color'] = FC

    def setColor(self, C):
        FC = np.zeros((2 * self.P0.shape[0], 4))
        FC[:, :] = C
        self.lines['color'] = FC

    def setMatrix(self, key, mat):
        self.lines[key] = mat

    def setModelMatrix(self, model):
        self.setMatrix('u_model', model)

    def setViewMatrix(self, view):
        self.setMatrix('u_view', view)

    def setProjectionMatrix(self, projection):
        self.setMatrix('u_projection', projection)

    def draw(self, mode=gl.GL_LINES):
        view = np.array(self.lines['u_view'], dtype=np.float32).reshape(4, 4)
        P0 = self.P0
        P1 = self.P1
        P0_offset = P0 - 0.01 * (np.linalg.inv(view[:3, :3])) @ np.array([0.0, 0.0, -1.0])
        P1_offset = P1 - 0.01 * (np.linalg.inv(view[:3, :3])) @ np.array([0.0, 0.0, -1.0])

        FV = np.hstack((P0_offset, P1_offset))
        FV = FV.reshape(-1, 3)
        self.FV = FV

        self.lines['position'] = FV

        gl.glLineWidth(self.W)
        self.lines.draw(mode)
