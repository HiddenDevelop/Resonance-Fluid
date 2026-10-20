#version 330 core
in vec2 uv;
out vec4 fragmentColor;
uniform sampler2D fluidTexture;
void main() {
    vec3 dye = max(texture(fluidTexture, uv).rgb, vec3(0.0));
    vec3 mapped = 1.0 - exp(-1.25 * dye);
    // Explicit display gamma; the framebuffer is not configured for sRGB conversion.
    fragmentColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}