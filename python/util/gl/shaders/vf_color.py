# -*- coding: utf-8 -*-
vertex = """
    uniform mat4   u_model;
    uniform mat4   u_view;
    uniform mat4   u_projection;
    uniform mat3   u_normal_mat;
    attribute vec3 position;
    attribute vec3 vector;
    varying vec3  v_vector;
    void main()
    {
        gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
        v_vector = vector;
    }
"""

fragment = """
    varying vec3  v_vector;
    void main()
    {
        vec3 color = 0.5 * normalize(v_vector) + vec3(0.5, 0.5, 0.5);
        gl_FragColor = vec4(color, 1.0);
    }
"""