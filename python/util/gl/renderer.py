import logging

import imageio
import matplotlib.pyplot as plt
import numpy as np
from glumpy import app, gl, glm, gloo, log

import util.gl.shaders.front_light as front_light_shader
import util.gl.shaders.mesh as mesh_shader
import util.gl.shaders.points as points_shader
from util.blender_info import BlenderInfo
from util.gl.lines import GLLines
from util.gl.mesh import GLMesh
from util.gl.picker import decode_picker
from util.gl.points import GLPoints

# ignore glumpy info log.
log.setLevel(logging.ERROR)

win_width, win_height = 512, 512
window = app.Window(win_width, win_height, color=(0.2, 0.2, 0.2, 1), visible=False)

texture = None
depth_buffer = None
framebuffer = None

meshes = []
lines = []
points = []

out_image = None


class Renderer:
    def __init__(self, im_width=1024, im_height=1024, is_picker=False):
        self.meshes = []
        self.lines = []
        self.points = []
        self.is_picker = is_picker

        global texture, depth_buffer, framebuffer

        if is_picker:
            texture = np.zeros((im_height, im_width, 4), np.ubyte).view(gloo.Texture2D)
            texture.interpolation = gl.GL_NEAREST
        else:
            texture = np.zeros((im_height, im_width, 4), np.float32).view(gloo.TextureFloat2D)

        depth_buffer = gloo.DepthBuffer(im_width, im_height, format=gl.GL_DEPTH_COMPONENT)
        framebuffer = gloo.FrameBuffer(color=[texture], depth=depth_buffer)

    def render(self, out_file=None, show_result=False, visible=False):
        global meshes, lines, points
        meshes = self.meshes
        lines = self.lines
        points = self.points

        app.run(framecount=0)

        if visible:
            window.show()

        global out_image
        if out_file is not None:
            if self.is_picker:
                imageio.imwrite(out_file, out_image)
            else:
                imageio.imwrite(out_file, np.uint8(255 * out_image))

        if show_result:
            fig_image = out_image
            fig = plt.figure()
            if self.is_picker:
                fig_image = decode_picker(out_image)
                plt.imshow(fig_image)
                plt.colorbar()
            else:
                plt.imshow(fig_image)
            plt.show()

        app.quit()

        return out_image

    def add_mesh(self, V, F, C=None, shader=mesh_shader):
        mesh = GLMesh(V, F, C=C, shader=shader)
        self.meshes.append(mesh)

    def add_lines(self, P0, P1, W=2.0, C=None, shader=mesh_shader):
        line = GLLines(P0, P1, W, C, shader)
        self.lines.append(line)

    def add_points(self, P, R, C=None, shader=points_shader):
        points = GLPoints(P, R, C, shader)
        self.points.append(points)

    def setMVPMat(self, model, view, projection):
        self.setModelMatrix(model)
        self.setViewMatrix(view)
        self.setProjectionMatrix(projection)

    def setMatrix(self, key, mat):
        for mesh in self.meshes:
            mesh.setMatrix(key, mat)

        for line in self.lines:
            line.setMatrix(key, mat)

        for point in self.points:
            point.setMatrix(key, mat)

    def setModelMatrix(self, model):
        self.setMatrix('u_model', model)

    def setViewMatrix(self, view):
        self.setMatrix('u_view', view)

    def setProjectionMatrix(self, projection):
        self.setMatrix('u_projection', projection)


def draw_objects():
    global meshes, lines, points

    for mesh in meshes:
        mesh.draw()

    for line in lines:
        line.draw()

    for point in points:
        point.draw()


@window.event
def on_draw(dt):
    global meshes, lines

    gl.glEnable(gl.GL_DEPTH_TEST)
    window.clear()
    framebuffer.activate()

    gl.glViewport(0, 0, framebuffer.width, framebuffer.height)

    gl.glClearColor(0, 0, 0, 0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    draw_objects()

    framebuffer.deactivate()
    gl.glViewport(0, 0, window.width, window.height)
    draw_objects()

    global out_image
    out_image = np.flipud(framebuffer.color[0].get())


@window.event
def on_init():
    gl.glEnable(gl.GL_LINE_SMOOTH)

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LEQUAL)
    gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
    gl.glEnable(gl.GL_POLYGON_OFFSET_POINT)
    gl.glPolygonOffset(-1.0, -1.0)
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
    gl.glEnable(gl.GL_POINT_SPRITE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)