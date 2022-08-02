# -*- coding: utf-8 -*-
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