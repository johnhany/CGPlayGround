#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // Unlike gl_Position in the vertex shader, there is no built-in variable to output a color for the current fragment. 
    // You have to specify your own output variable for each framebuffer where the layout(location = 0) modifier specifies the index of the framebuffer
    outColor = vec4(fragColor, 1.0);
}