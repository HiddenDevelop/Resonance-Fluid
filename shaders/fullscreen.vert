#version 330 core
out vec2 uv;
void main() {
    // One oversized triangle covers the viewport without an internal seam.
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}