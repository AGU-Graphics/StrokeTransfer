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
        v_normal = u_normal_mat * normal;
        v_color = color;
    }
"""

fragment = """
    varying vec4  v_color;
    varying vec3  v_normal;
    void main()
    {
        vec3 normal = normalize(v_normal);
        float I = clamp(0.8*normal.z+0.2, 0.0, 1.0);
        vec3 color = I * v_color.rgb;
        gl_FragColor = vec4(color, v_color.a);
    }
"""