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
import seaborn as sns
from matplotlib import cm, pyplot as plt


# sns.set()

def image_points(width, height):
    xs = range(width)
    ys = range(height)
    X, Y = np.meshgrid(xs, ys)
    P = np.dstack((X, Y))
    return P


def font_setting(font_family='Times New Roman', font_size=12):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = font_family
    plt.rcParams["font.size"] = font_size


def get_color_palette(n_colors=24):
    return sns.color_palette(n_colors=n_colors)
    # return get_color_list()


def save_fig(file_path, bbox_inches="tight", pad_inches=0.05, transparent=False):
    fig = plt.gcf()
    fig.savefig(file_path, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent)


def im_crop(ax, I, xlim, ylim):
    h, w = I.shape[:2]
    ax.set_xlim(np.array(xlim) * w)
    ax.set_ylim(np.array((ylim[1], ylim[0])) * h)


def cmap_for_scalar():
    return cm.magma


def get_color_list():
    color_list = np.array([[93, 245, 235],
                           [128, 94, 214],
                           [235, 134, 126],
                           [245, 186, 23],
                           [107, 250, 84]]) / 255.0
    return color_list


def plot_image(I, cmap=cmap_for_scalar(), vmin=None, vmax=None):
    ax = plt.gca()

    img_plt = plt.imshow(I, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_facecolor([1.0, 1.0, 1.0])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.axis("off")
    return img_plt


def draw_bg(A, bg_color=[1.0, 1.0, 1.0]):
    h, w = A.shape[:2]
    C = np.zeros((h, w, 4))

    for ci in range(3):
        C[:, :, ci] = bg_color[ci]

    C[:, :, 3] = 1.0 - A
    plt.imshow(C)


def plot_vf_grid(V, s=20, color="red", scale=1.0, width_scale=1.0):
    if V is None:
        return
    P = image_points(V.shape[1], V.shape[0])

    Ps = P[::s, ::s, :]
    Vs = V[::s, ::s, :]

    ids = Vs[:, :, 3] > 0.5

    Ps = Ps[ids, :]
    Vs = Vs[ids, :]

    plt.quiver(Ps[:, 0], Ps[:, 1], scale * Vs[:, 0],
               scale * Vs[:, 1], color=color, angles='xy',
               scale=100.0, width=0.005 * width_scale)


def vf_show(V):
    V_color = 0.5 * V + 0.5
    V_color[:, :, 3] = V[:, :, 3]
    V_color = np.clip(V_color, 0.0, 1.0)
    plot_image(V_color)


def title_under(ax=None, title="", offset=-0.05):
    if ax is None:
        ax = plt.gca()
    ax.text(0.5, offset, title, va='top', ha='center', transform=ax.transAxes)


def title_vertical(ax=None, title=""):
    if ax is None:
        ax = plt.gca()
    ax.text(-0.03, 0.5, title,
            horizontalalignment='right',
            verticalalignment='center',
            rotation=270,
            transform=ax.transAxes)


font_setting()
