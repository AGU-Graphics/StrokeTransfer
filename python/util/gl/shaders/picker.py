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


vertex = """
    uniform mat4   u_model;
    uniform mat4   u_view;
    uniform mat4   u_projection;
    uniform mat3   u_normal_mat;
    attribute vec3 position;
    attribute float id;
    varying vec4  v_id;
    void main()
    {
        gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
        v_id = vec4 ( mod(floor(id / (256*256)), 256) / 255.0,
                      mod(floor(id /     (256)), 256) / 255.0,
                      mod(floor(id /       (1)), 256) / 255.0,
                      1.0 );
    }
"""

fragment = """
    varying vec4  v_id;
    void main()
    {
        gl_FragColor = v_id;
    }
"""