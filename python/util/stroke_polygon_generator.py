import numpy as np

from glumpy import gl, glm, gloo
from PIL import Image

from util.normalize.normalize_position import normalize_positions01
from util.verbose import verbose_range


vertex = """
    uniform mat4   u_model;
    uniform mat4   u_view;
    uniform mat4   u_projection;
    attribute vec3 position;
    attribute vec3 normal;
    varying vec3  v_normal;
    attribute vec4 color;
    varying vec4  v_color;
    attribute vec2 tex_coord;
    varying vec2 uv;
    void main()
    {
        gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
        uv = tex_coord;
    }
"""

fragment = """
    uniform vec4 color;
    uniform sampler2D texture;
    varying vec2 uv;
    void main()
    {
        vec4 texel = texture2D(texture, uv);
        if(texel.a<0.5)
            discard;
        gl_FragColor = color * texel; // texture2D(texture, uv);
    }
"""


def norm_vectors(V):
    return np.sqrt(np.einsum("ij,ij->i", V, V))


def normalize_vectors(V):
    V_norms = norm_vectors(V)
    W = 1.0 / V_norms
    V_normalized = np.einsum("i, ij->ij", W, V)
    return V_normalized


def vertex_project(camera_info, P):
    model_mat, view_mat, project_mat = camera_info.MVPMatrix()
    MVP = project_mat.T @ view_mat.T @ model_mat.T
    P_proj = np.hstack([P, np.ones((P.shape[0], 1))]) @ MVP.T
    for i in range(3):
        P_proj[:, i] = (P_proj[:, i] / P_proj[:, 3]) / 2 + 0.5
    return P_proj


def stroke_polygon_generator(integral_curve_datas, image_data, strokes, drawing_stroke_num, camera_info, texture_file_name, verbose, half_on):
    """

    Args:
        integral_curve_datas: integral curve data.
        image_data: image data.
        strokes: stroke data already processed.
        drawing_stroke_num: number of drawing strokes. if the value is 0, all strokes will be drawn.
        camera_info: camera info exported from blender.
        texture_file_name: brush texture file name.
        verbose: if true, print intermediate info for the command.
        half_on: if True, the half width is used for stroke drawing.

    Returns:
        strokes: all stroke data that will be drawn.
    """
    epsilon = 1e-4
    model_mat, view_mat, project_mat = camera_info.MVPMatrix()

    if drawing_stroke_num > len(integral_curve_datas.lines) or drawing_stroke_num == 0:
        len_stroke = len(integral_curve_datas.lines)
    else:
        len_stroke = drawing_stroke_num
    for k in verbose_range(verbose, range(len_stroke)):
        P = np.array(integral_curve_datas.lines[k])
        N = np.array(integral_curve_datas.lines_normals[k])
        PN = P + epsilon * N

        width = np.array(integral_curve_datas.widths)[k]
        C = np.array(integral_curve_datas.colors[k])

        if half_on:
            quarter_plot_num = int(np.floor(PN.shape[0] / 4))
            PN = PN[quarter_plot_num:len(PN) - quarter_plot_num]
            N = N[quarter_plot_num:len(N) - quarter_plot_num]
            width = width / 2

        stroke = Stroke(image_data, PN, N, width, C, texture_file_name, camera_info)
        stroke.setMVPMat(model_mat, view_mat, project_mat)
        strokes.append(stroke)

    return strokes


class Stroke:
    """Stroke data definition.

    Attributes:
        positions: (n, 3) position data.
        normals: (n, 3) normal data.
        width: (n, 1) width array.
        color: (4) color info.
        texture_file_name: brush texture file name.
        strip: triangle strip data for stroke.
    """

    def __init__(self, image_data, positions, normals, width, color, texture_file_name, camera_info):
        """
        Args:
            image_data: image data.
            positions: (n, 3) position data.
            normals: (n, 3) normal data.
            width: (n, 1) width array.
            color: (4) color info.
            texture_file_name: brush texture file name.
            camera_info: camera info exported from blender.
        """
        self.positions = np.array(positions)
        self.normals = np.array(normals)
        self.width = width
        self.color = color
        self.tex_file = texture_file_name
        self.strip = self.gen_strip(image_data, camera_info)

        self.strip["u_model"] = np.eye(4, dtype=np.float32)
        self.strip["u_view"] = glm.translation(0, 0, 0.5)
        self.strip["u_projection"] = np.eye(4, dtype=np.float32)

    def draw(self):
        self.strip.draw(gl.GL_TRIANGLE_STRIP)

    def compute_frame(self):
        P = self.positions
        N = self.normals

        E = P[1:, :] - P[:-1, :]
        E_normalized = normalize_vectors(E)
        T = np.zeros_like(P)
        T[1:-1, :] = 0.5 * (E_normalized[1:, :] + E_normalized[:-1, :])
        T[0, :] = E_normalized[0, :]
        T[-1, :] = E_normalized[-1, :]
        T = normalize_vectors(T)

        B = np.cross(T, N)
        B = normalize_vectors(B)

        T = np.cross(N, B)
        return T, B

    def compute_arc_parameter(self):
        P = self.positions
        E = P[1:, :] - P[:-1, :]
        len_strokes = np.sqrt(np.sum(E ** 2, axis=1))
        arc_parameter = np.zeros((P.shape[0]))
        arc_parameter[1:] = np.cumsum(len_strokes)
        arc_parameter /= arc_parameter[-1]
        return arc_parameter

    def width_screen(self, image_data, camera_info):
        P = self.positions
        W = self.width

        im_width = image_data.img_width
        im_height = image_data.img_height

        if isinstance(W, float):
            W = W * np.ones((P.shape[0]))
        W_half = W / 2.0

        T, B = self.compute_frame()

        WB = np.einsum("i,ij->ij", W_half, B)
        P_minus_WB_proj = vertex_project(camera_info, P - WB)
        P_minus_WB_proj = normalize_positions01(P_minus_WB_proj, im_width, im_height)
        P_plus_WB_proj = vertex_project(camera_info, P + WB)
        P_plus_WB_proj = normalize_positions01(P_plus_WB_proj, im_width, im_height)
        W_screen = np.sqrt((P_minus_WB_proj[:, 0] - P_plus_WB_proj[:, 0]) ** 2 + (P_minus_WB_proj[:, 1] - P_plus_WB_proj[:, 1]) ** 2)
        W = W * np.clip(W / W_screen, None, 100) / 2
        WB = np.einsum("i,ij->ij", W, B)
        return WB

    def gen_strip(self, image_data, camera_info):
        P = self.positions

        arc_parameter = self.compute_arc_parameter()

        U1 = np.dstack((arc_parameter, np.zeros_like(arc_parameter))).reshape(P.shape[0], -1)
        U2 = np.dstack((arc_parameter, np.ones_like(arc_parameter))).reshape(P.shape[0], -1)

        U = np.hstack((U1, U2)).reshape(2 * P.shape[0], -1)

        WB = self.width_screen(image_data, camera_info)
        strip_positions = np.hstack((P - WB, P + WB)).reshape(2 * P.shape[0], -1)
        strip_tex_coord = U

        strip = gloo.Program(vertex, fragment, count=2 * P.shape[0])
        strip["position"] = strip_positions
        strip["tex_coord"] = strip_tex_coord
        strip["texture"] = Image.open(self.tex_file)
        strip["color"] = self.color
        return strip

    def setMVPMat(self, model, view, projection):
        self.setModelMatrix(model)
        self.setViewMatrix(view)
        self.setProjectionMatrix(projection)

    def setModelMatrix(self, model):
        self.strip["u_model"] = model

    def setViewMatrix(self, view):
        self.strip["u_view"] = view

    def setProjectionMatrix(self, projection):
        self.strip["u_projection"] = projection
