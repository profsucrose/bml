#version 330
out vec4 FragColor;

uniform vec2 coord;
uniform vec2 resolution;
uniform float frame;
uniform float frame_count;

void main() {
    vec2 uv = coord / resolution

    float distance = dist(uv, [0.5, 0.5])

    vec3 color = sin(uv + frame / frame_count)

    return [color.xy, 1, 1]
}