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


import igl
import numpy as np
from glumpy import gl, glm, gloo

import util.gl.shaders.mesh as mesh_shader
import util.gl.shaders.picker as picker_shader
from util.gl.lines import GLLines


class GLMesh:
    def __init__(self, V, F, C=None, shader=mesh_shader):
        self.F = F
        self.C = C

        FV = V[F]
        FV = FV.reshape(-1, 3)

        mesh = gloo.Program(shader.vertex, shader.fragment, count=FV.shape[0])
        self.mesh = mesh

        self.vf = {}

        self.setPosition(V)

        mesh['u_model'] = np.eye(4, dtype=np.float32)
        mesh['u_view'] = glm.translation(0, 0, 0.5)
        projection = np.eye(4, dtype=np.float32)
        mesh['u_projection'] = projection
        mesh['u_normal_mat'] = np.eye(3, dtype=np.float32)

        if shader == picker_shader:
            F_id = np.arange(len(F), dtype=np.float32)
            F_id = np.vstack((F_id, F_id, F_id)).T.flatten()
            mesh['id'] = F_id
            return

        N_v = igl.per_vertex_normals(V, F)
        self.setNormal(N_v)

        if C is not None:
            if len(C) <= 4:
                self.setColor(C)
            elif C.shape[0] == V.shape[0]:
                self.setVertexColor(C)
            else:
                self.setFaceColor(C)
        else:
            self.setVertexColor(0.5 * N_v + 0.5)

        self.vf['normal'] = GLLines(V, V, C=np.array([1.0, 0.0, 0.0, 1.0]))
        self.vf['normal'].setVector(N_v, scale=0.025)

    def add_vector_field(self, name, C=np.array([1.0, 0.0, 0.0, 1.0])):
        V = self.V
        self.vf[name] = GLLines(V, V, C=C)

    def set_orientation(self, orientation):
        self.setNormal(orientation)
        # self.mesh['orientation'] = orientation

    def setMatrix(self, key, mat):
        self.mesh[key] = mat

        for vf_key in self.vf.keys():
            self.vf[vf_key].lines[key] = mat

    def setModelMatrix(self, model):
        self.setMatrix('u_model', model)

    def setViewMatrix(self, view):
        self.setMatrix('u_view', view)

    def setProjectionMatrix(self, projection):
        self.setMatrix('u_projection', projection)

    def setPosition(self, V):
        self.V = V
        F = self.F
        FV = V[F]
        FV = FV.reshape(-1, 3)
        self.mesh['position'] = FV

        for key in self.vf.keys():
            self.vf[key].setP0(V)

    def setNormal(self, N):
        self.N = N
        F = self.F
        FN = N[F]
        FN = FN.reshape(-1, 3)
        self.mesh['normal'] = FN

    def color_with_alpha(self, C):
        C2 = np.ones((C.shape[0], 4))
        if C.size == C.shape[0]:
            for ci in range(3):
                C2[:, ci] = C
            return C2

        if C.shape[1] == 3:
            C2[:, :3] = C
            return C2
        return C

    def setColor(self, c):
        C = np.ones((self.V.shape[0], 4))
        for ci in range(len(c)):
            C[:, ci] = c[ci]
        self.setVertexColor(C)

    def setVertexColor(self, C):
        C2 = self.color_with_alpha(C)
        self.C = C2
        F = self.F
        FC = C2[F]

        FC = FC.reshape(-1, 4)
        self.mesh['color'] = FC

    def setFaceColor(self, C):
        C2 = self.color_with_alpha(C)
        F = self.F
        FC = np.hstack((C2, C2, C2)).reshape(-1, 4)
        self.mesh['color'] = FC

    def draw(self, mode=gl.GL_TRIANGLES):
        model = np.array(self.mesh['u_model'], dtype=np.float32).reshape(4, 4)
        view = np.array(self.mesh['u_view'], dtype=np.float32).reshape(4, 4)

        MV = model @ view
        self.mesh['u_normal_mat'] = MV[:3, :3]

        self.mesh.draw(mode)

    def drawEdge(self, mode=gl.GL_TRIANGLES):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.mesh.draw(mode)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
