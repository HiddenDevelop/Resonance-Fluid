#include "Input.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
namespace {
bool pressed(GLFWwindow* w,int key,bool& before) {
    bool now=glfwGetKey(w,key)==GLFW_PRESS, edge=now&&!before; before=now; return edge;
}
}
void Input::poll(GLFWwindow* w,double frameSeconds,int n) {
    pausePressed=pressed(w,GLFW_KEY_SPACE,previousPause_);
    resetPressed=pressed(w,GLFW_KEY_R,previousReset_);
    colorPressed=pressed(w,GLFW_KEY_C,previousColor_);
    if(colorPressed) color_=(color_+1)%3;
    decayAdjustment=float(frameSeconds)*0.1f*((glfwGetKey(w,GLFW_KEY_UP)==GLFW_PRESS)-(glfwGetKey(w,GLFW_KEY_DOWN)==GLFW_PRESS));
    if(glfwGetKey(w,GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(w,GLFW_TRUE);
    int width,height; glfwGetWindowSize(w,&width,&height);
    double mx,my; glfwGetCursorPos(w,&mx,&my);
    int side=std::min(width,height);
    if(side<=0) { dragging_=false; discardMotion(); return; }
    float nx=float((mx-(width-side)*0.5)/side), ny=1-float((my-(height-side)*0.5)/side);
    bool active=glfwGetWindowAttrib(w,GLFW_FOCUSED) && glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS && nx>=0 && nx<=1 && ny>=0 && ny<=1;
    if(active && dragging_) {
        // Integrated cursor displacement, consumed once even when rendering outruns simulation.
        pendingX_+=(nx-previousX_)*n*7.0f;
        pendingY_+=(ny-previousY_)*n*7.0f;
    }
    if(!active) discardMotion();
    dragging_=active; previousX_=x_=nx; previousY_=y_=ny;
}
void Input::inject(FluidSimulation& simulation,float dt) {
    if(!dragging_) return;
    const float3 colors[]={{0.03f,0.65f,1.0f},{0.5f,0.06f,1.0f},{1.0f,0.03f,0.4f}};
    Splat s; s.x=x_; s.y=y_; s.color=colors[color_]; s.radius=simulation.size()*0.027f;
    // Bound a single impulse after a rapid cursor jump. No division by tiny frame time.
    float magnitude=std::hypot(pendingX_,pendingY_);
    float scale=magnitude>180 ? 180/magnitude : 1;
    s.velocityX=pendingX_*scale; s.velocityY=pendingY_*scale; s.dyeAmount=12*dt;
    simulation.addSplat(s); discardMotion();
}
