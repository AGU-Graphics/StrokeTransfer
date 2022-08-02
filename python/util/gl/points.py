# -*- coding: utf-8 -*-
import numpy as np
from glumpy import gl, glm, gloo

import util.gl.shaders.points as points_shader


class GLPoints:
    def __init__(self, P, R=5, C=None, shader=points_shader):
        self.P = P
        self.R = R
        self.C = C

        points = gloo.Program(shader.vertex, shader.fragment, count=P.shape[0])
        self.points = points

        points['position'] = P

        if C is not None:
            self.setColor(C)

        points['u_model'] = np.eye(4, dtype=np.float32)
        points['u_view'] = glm.translation(0, 0, 0.5)
        projection = np.eye(4, dtype=np.float32)
        points['u_projection'] = projection
        points['u_normal_mat'] = np.eye(3, dtype=np.float32)

    def setMatrix(self, key, mat):
        self.points[key] = mat

    def setModelMatrix(self, model):
        self.setMatrix('u_model', model)

    def setViewMatrix(self, view):
        self.setMatrix('u_view', view)

    def setProjectionMatrix(self, projection):
        self.setMatrix('u_projection', projection)

    def setColors(self, C):
        FC = np.array([[Ci, Ci] for Ci in C])
        FC = FC.reshape(-1, 4)
        self.points['color'] = FC

    def setColor(self, C):
        FC = np.zeros((self.P.shape[0], 4))
        FC[:, :] = C
        self.points['color'] = FC

    def draw(self, mode=gl.GL_POINTS):
        view = np.array(self.points['u_view'], dtype=np.float32).reshape(4, 4)

        self.points['position'] = self.P - 0.01 * (np.linalg.inv(view[:3, :3])) @ np.array([0.0, 0.0, -1.0])

        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_POINT_SPRITE)
        gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glPointSize(self.R)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glColorMask(True, True, True, False)
        self.points.draw(mode)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glColorMask(True, True, True, True)
