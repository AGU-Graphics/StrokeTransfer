# -*- coding: utf-8 -*-
vertex = """
    uniform mat4   u_model;
    uniform mat4   u_view;
    uniform mat4   u_projection;
    uniform mat3   u_normal_mat;
    attribute vec3 position;
    attribute vec3 normal;
    varying vec3  v_normal;
    attribute vec4 color;
    varying vec4  v_color;
    void main()
    {
        gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
        mat3 normal_mat = mat3(u_view * u_model);
        v_normal = u_normal_mat * normal;
        v_color = color;
    }
"""

fragment = """
    varying vec4  v_color;
    varying vec3  v_normal;
    void main()
    {
        vec3 color = 0.5 * normalize(v_normal) + vec3(0.5, 0.5, 0.5);
        gl_FragColor = vec4(color, 1.0);
    }
"""