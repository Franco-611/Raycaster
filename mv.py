#Para rotar en el eje X se utilizan las flechas Derecha e Izquierda.
#Para cambiar de shader se utilizan los numeros del 1 al 3.

import pygame
import glm
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import * 
from Obj import *

pygame.init()

screen = pygame.display.set_mode(
    (800, 800),
    pygame.OPENGL | pygame.DOUBLEBUF
)

vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertexColor;

uniform mat4 matrix;

out vec3 ourColor;
out vec2 fragCoord;
void main()
{
    gl_Position = matrix * vec4(position, 1.0f);
    fragCoord =  gl_Position.xy;
    ourColor = vertexColor;
}
"""

fragment_shader = """
#version 460
layout (location = 0) out vec4 fragColor;

uniform vec3 color;
uniform float iTime;

in vec3 ourColor;
in vec2 fragCoord;

// Variable to a keep a copy of the noise value prior to palettization. Used to run a soft gradient 
// over the surface, just to break things up a little.
float ns;


//float sFract(float x, float sm){ float fx = fract(x); return fx - smoothstep(fwidth(x)*sm, 0., 1. - fx); }
//float sFract(float x, float sm){ float fx = fract(x); return min(fx, fx*(1. - fx)/fwidth(x)/sm); }

// Based on Ollj's smooth "fract" formula.
float sFract(float x, float sm){
    
    // Extra smoothing factor. "1" is the norm.
    const float sf = 1.; 
    
    // The hardware "fwidth" is cheap, but you could take the expensive route and
    // calculate it by hand if more quality was required.
    vec2 u = vec2(x, fwidth(x)*sf*sm);
    
    // Ollj's original formula with a transcendental term omitted.
    u.x = fract(u.x);
    u += (1. - 2.*u)*step(u.y, u.x);
    return clamp(1. - u.x/u.y, 0., 1.); // Cos term ommitted.
}



// Only correct for nonnegative values, but in this example, numbers aren't negative.
float sFloor(float x){ return x - sFract(x, 1.); } 

// Standard hue rotation formula with a bit of streamlining. 
vec3 rotHue(vec3 p, float a){

    vec2 cs = sin(vec2(1.570796, 0) + a);

    mat3 hr = mat3(0.299,  0.587,  0.114,  0.299,  0.587,  0.114,  0.299,  0.587,  0.114) +
        	  mat3(0.701, -0.587, -0.114, -0.299,  0.413, -0.114, -0.300, -0.588,  0.886) * cs.x +
        	  mat3(0.168,  0.330, -0.497, -0.328,  0.035,  0.292,  1.250, -1.050, -0.203) * cs.y;
							 
    return clamp(p*hr, 0., 1.);
}


/*
// Fabrices concise, 2D rotation formula.
mat2 r2(float th){ vec2 a = sin(vec2(1.5707963, 0) + th); return mat2(a, -a.y, a.x); }

// Dave's hash function. More reliable with large values, but will still eventually break down.
//
// Hash without Sine
// Creative Commons Attribution-ShareAlike 4.0 International Public License
// Created by David Hoskins.
// vec3 to vec3.
vec3 hash33(vec3 p){

	p = fract(p * vec3(.1031, .1030, .0973));
    p += dot(p, p.yxz + 19.19);
    p = fract((p.xxy + p.yxx)*p.zyx)*2. - 1.;
    return p;
    
    // Note the "mod" call. Slower, but ensures accuracy with large time values.
    //mat2  m = r2(mod(iTime*2., 6.2831853));	
	//p.xy = m * p.xy;//rotate gradient vector
    //p.yz = m * p.yz;//rotate gradient vector
    //p.xz = m * p.xz;//rotate gradient vector
    
    //mat3 m = r3(mod(iTime*2., 6.2831853));	
    //vec3 th = mod(vec3(.31, .53, .97) + iTime*2., 6.2831853);
    //mat3 m = r3(th.x, th.y, th.z);
    //p *= m;
	return p;

}
*/

// vec3 to vec3 hash algorithm.
vec3 hash33(vec3 p){ 

    // Faster, but doesn't disperse things quite as nicely as the block below it. However, when framerate
    // is an issue, and it often is, this is the one to use. Basically, it's a tweaked amalgamation I put
    // together, based on a couple of other random algorithms I've seen around... so use it with caution,
    // because I make a tonne of mistakes. :)
    float n = sin(dot(p, vec3(7, 157, 113)));    
    return fract(vec3(2097152, 262144, 32768)*n)*2. - 1.; // return fract(vec3(64, 8, 1)*32768.0*n)*2.-1.; 

    // I'll assume the following came from IQ.
    //p = vec3( dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), 
    //          dot(p, vec3(113.5, 271.9, 124.6)));
    //return (fract(sin(p)*43758.5453)*2. - 1.);

}



// Cheap, streamlined 3D Simplex noise... of sorts. I cut a few corners, so it's not perfect, but it's
// artifact free and does the job. I gave it a different name, so that it wouldn't be mistaken for
// the real thing.
// 
// Credits: Ken Perlin, the inventor of Simplex noise, of course. Stefan Gustavson's paper - 
// "Simplex Noise Demystified," IQ, other "ShaderToy.com" people, etc.
float tetraNoise(in vec3 p){

    // Skewing the cubic grid, then determining the first vertice and fractional position.
    vec3 i = floor(p + dot(p, vec3(1./3.)) );  p -= i - dot(i, vec3(1./6.));
    
    // Breaking the skewed cube into tetrahedra with partitioning planes, then determining which side of the 
    // intersecting planes the skewed point is on. Ie: Determining which tetrahedron the point is in.
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1. - i1.zxy); i1 = min(i1, 1. - i1.zxy);    
    
    // Using the above to calculate the other three vertices -- Now we have all four tetrahedral vertices.
    // Technically, these are the vectors from "p" to the vertices, but you know what I mean. :)
    vec3 p1 = p - i1 + 1./6., p2 = p - i2 + 1./3., p3 = p - .5;
  

    // 3D simplex falloff - based on the squared distance from the fractional position "p" within the 
    // tetrahedron to the four vertice points of the tetrahedron. 
    vec4 v = max(.5 - vec4(dot(p, p), dot(p1, p1), dot(p2, p2), dot(p3, p3)), 0.);
    
    // Dotting the fractional position with a random vector, generated for each corner, in order to determine 
    // the weighted contribution distribution... Kind of. Just for the record, you can do a non-gradient, value 
    // version that works almost as well.
    vec4 d = vec4(dot(p, hash33(i)), dot(p1, hash33(i + i1)), dot(p2, hash33(i + i2)), dot(p3, hash33(i + 1.)));
     
     
    // Simplex noise... Not really, but close enough. :)
    return clamp(dot(d, v*v*v*8.)*1.732 + .5, 0., 1.); // Not sure if clamping is necessary. Might be overkill.

}


// The function value. In this case, slightly-tapered, quantized Simplex noise.
float func(vec2 p){
    
    // The noise value.
    float n = tetraNoise(vec3(p.x*4., p.y*4., 0) - vec3(0, .25, .5)*iTime);
    
    // A tapering function, similar in principle to a smooth combine. Used to mutate or shape 
    // the value above. This one tapers it off into an oval shape and punches in a few extra holes.
    // Airtight uses a more interesting triangular version in his "Cartoon Fire" shader.
    float taper = .1 + dot(p, p*vec2(.35, 1));
	n = max(n - taper, 0.)/max(1. - taper, .0001);
    
    // Saving the noise value prior to palettization. Used for a bit of gradient highlighting.
    ns = n; 
    
    // I remember reasoning to myself that the following would take a continuous function ranging
    // from zero to one, then palettize it over "palNum" discreet values between zero and one
    // inclusive. It seems to work, but if my logic is lacking (and it often is), feel free to 
    // let me know. :)
    const float palNum = 9.; 
    // The range should strictly fall between zero and one, but for some crazy reason, numbers fall
    // outside the range, so I've had to clamp it. I know the computer is never wrong, so I'm 
    // probably overlooking something. Having said that, I don't trust the GPU "fract" function much.
    //return clamp(sFloor(n*(palNum - .001))/(palNum - 1.), 0., 1.);
    return n*.25 + clamp(sFloor(n*(palNum - .001))/(palNum - 1.), 0., 1.)*.75;
    
}



void main()
{
    vec2 iResolution = vec2(6, 6);
    // Screen coordinates.
	vec2 u = (fragCoord.xy - iResolution.xy*.5)/iResolution.y;
    
    // Function value.
    float f = func(u);
    float ssd = ns; // Saving the unpalettized noise value to add a little gradient to the color, etc.
    
    // Four sample values around the original. Used for edging and highlighting.
    vec2 e = vec2(1.5/iResolution.y, 0);
    float fxl = func(u + e.xy);
    float fxr = func(u - e.xy);
    float fyt = func(u + e.yx);
    float fyb = func(u - e.yx);
    
    // Colorizing the function value, and applying some hue rotation based on position.
    // Most of it was made up.
    vec3 col = pow(min(vec3(1.5, 1, 1)*(f*.7 + ssd*.35), 1.), vec3(1, 2., 10)*2.) + .01;
    col = rotHue(col, -.25+.4*length(u));

    // Applying the dark edges.
    col *= max(1. - (abs(fxl - fxr) + abs(fyt - fyb))*5., 0.);
    //col *= max(1. - length(vec2(fxl, fyt) - vec2(fxr, fyb))*7., 0.);
    // Resampling with a slightly larger spread to provide some highlighting.
    fxl = func(u + e.xy*1.5);
    fyt = func(u + e.yx*1.5);
    col += vec3(.5, .7, 1)*(max(f - fyt, 0.) + max(f - fxl, 0.))*ssd*10.;
    
    // Subtle, bluish vignette.
    //u = fragCoord/iResolution.xy;
    //col = mix(vec3(0, .1, 1), col, pow( 16.0*u.x*u.y*(1.0-u.x)*(1.0-u.y) , .125)*.15 + .85);

 	
    // Rough gamma correction.
    fragColor = vec4(sqrt(clamp(col, 0., 1.)), 1);
    
}
"""

f_shader_2 = """
#version 460
layout (location = 0) out vec4 fragColor;

uniform vec3 color;
uniform float iTime;

in vec3 ourColor;
in vec2 fragCoord;

// This is my favorite fire palette. It's trimmed down for shader usage, and is based on an 
// article I read at Hugo Elias's site years ago. I'm sure most old people, like me, have 
// visited his site at one time or another:
//
// http://freespace.virgin.net/hugo.elias/models/m_ffire.htm
//
vec3 firePalette(float i){

    float T = 1400. + 1300.*i; // Temperature range (in Kelvin).
    vec3 L = vec3(7.4, 5.6, 4.4); // Red, green, blue wavelengths (in hundreds of nanometers).
    L = pow(L,vec3(5)) * (exp(1.43876719683e5/(T*L)) - 1.);
    return 1. - exp(-5e8/L); // Exposure level. Set to "50." For "70," change the "5" to a "7," etc.
}

/*
vec3 firePalette(float i){

    float T = 1400. + 1300.*i; // Temperature range (in Kelvin).
    // Hardcode red, green and blue wavelengths (in hundreds of nanometers).
    vec3 L = (exp(vec3(19442.7999572, 25692.271372, 32699.2544734)/T) - 1.);
    // Exposure level. Set to "50" For "70," change the ".5" to a ".7," etc.
    return 1. - exp(-vec3(22532.6051122, 90788.296915, 303184.239775)*2.*.5/L); 
}
*/

// Hash function. This particular one probably doesn't disperse things quite as nicely as some 
// of the others around, but it's compact, and seems to work.
//
vec3 hash33(vec3 p){ 
    
    float n = sin(dot(p, vec3(7, 157, 113)));    
    return fract(vec3(2097152, 262144, 32768)*n); 
}

// 3D Voronoi: Obviously, this is just a rehash of IQ's original.
//
float voronoi(vec3 p){

	vec3 b, r, g = floor(p);
	p = fract(p); // "p -= g;" works on some GPUs, but not all, for some annoying reason.
	
	// Maximum value: I think outliers could get as high as "3," the squared diagonal length 
	// of the unit cube, with the mid point being "0.75." Is that right? Either way, for this 
	// example, the maximum is set to one, which would cover a good part of the range, whilst 
	// dispensing with the need to clamp the final result.
	float d = 1.; 
     
    // I've unrolled one of the loops. GPU architecture is a mystery to me, but I'm aware 
    // they're not fond of nesting, branching, etc. My laptop GPU seems to hate everything, 
    // including multiple loops. If it were a person, we wouldn't hang out. 
	for(int j = -1; j <= 1; j++) {
	    for(int i = -1; i <= 1; i++) {
    		
		    b = vec3(i, j, -1);
		    r = b - p + hash33(g+b);
		    d = min(d, dot(r,r));
    		
		    b.z = 0.0;
		    r = b - p + hash33(g+b);
		    d = min(d, dot(r,r));
    		
		    b.z = 1.;
		    r = b - p + hash33(g+b);
		    d = min(d, dot(r,r));
    			
	    }
	}
	
	return d; // Range: [0, 1]
}

// Standard fBm function with some time dialation to give a parallax 
// kind of effect. In other words, the position and time frequencies 
// are changed at different rates from layer to layer.
//
float noiseLayers(in vec3 p) {

    // Normally, you'd just add a time vector to "p," and be done with 
    // it. However, in this instance, time is added seperately so that 
    // its frequency can be changed at a different rate. "p.z" is thrown 
    // in there just to distort things a little more.
    vec3 t = vec3(0., 0., p.z + iTime*1.5);

    const int iter = 5; // Just five layers is enough.
    float tot = 0., sum = 0., amp = 1.; // Total, sum, amplitude.

    for (int i = 0; i < iter; i++) {
        tot += voronoi(p + t) * amp; // Add the layer to the total.
        p *= 2.; // Position multiplied by two.
        t *= 1.5; // Time multiplied by less than two.
        sum += amp; // Sum of amplitudes.
        amp *= .5; // Decrease successive layer amplitude, as normal.
    }
    
    return tot/sum; // Range: [0, 1].
}

void main()
{
    vec2 iResolution = vec2(1, 1);
    // Screen coordinates.
	vec2 uv = (fragCoord - iResolution.xy*.5) / iResolution.y;
	
	// Shifting the central position around, just a little, to simulate a 
	// moving camera, albeit a pretty lame one.
	uv += vec2(sin(iTime*.5)*.25, cos(iTime*.5)*.125);
	
    // Constructing the unit ray. 
	vec3 rd = normalize(vec3(uv.x, uv.y, 3.1415926535898/8.));

    // Rotating the ray about the XY plane, to simulate a rolling camera.
	float cs = cos(iTime*.25), si = sin(iTime*.25);
    // Apparently "r *= rM" can break in some older browsers.
	rd.xy = rd.xy*mat2(cs, -si, si, cs); 
	
	// Passing a unit ray multiple into the Voronoi layer function, which 
	// is nothing more than an fBm setup with some time dialation.
	float c = noiseLayers(rd*2.);
	
	// Optional: Adding a bit of random noise for a subtle dust effect. 
	c = max(c + dot(hash33(rd)*2. - 1., vec3(.015)), 0.);

    // Coloring:
    
    // Nebula.
    c *= sqrt(c)*1.5; // Contrast.
    vec3 col = firePalette(c); // Palettization.
    //col = mix(col, col.zyx*.1+ c*.9, clamp((1.+rd.x+rd.y)*0.45, 0., 1.)); // Color dispersion.
    col = mix(col, col.zyx*.15 + c*.85, min(pow(dot(rd.xy, rd.xy)*1.2, 1.5), 1.)); // Color dispersion.
    col = pow(col, vec3(1.25)); // Tweaking the contrast a little.
    
    // The fire palette on its own. Perhaps a little too much fire color.
    //c = pow(c*1.33, 1.25);
    //vec3 col =  firePalette(c);
   
    // Black and white, just to keep the art students happy. :)
	//c *= c*1.5;
	//vec3 col = vec3(c);
	
	// Rough gamma correction, and done.
	fragColor = vec4(sqrt(clamp(col, 0., 1.)), 1);
}
"""

f_shader_3 = """
#version 460
layout (location = 0) out vec4 fragColor;

uniform vec3 color;
uniform float iTime;

in vec3 ourColor;
in vec2 fragCoord;

// Glossy version. It's there to show that the method works with raised surfaces too.
//#define GLOSSY

// Standard 2x2 hash algorithm.
vec2 hash22(vec2 p) {
    
    // Faster, but probaly doesn't disperse things as nicely as other methods.
    float n = sin(dot(p, vec2(41, 289)));
    p = fract(vec2(2097152, 262144)*n);
    return cos(p*6.283 + iTime)*.5;
    //return abs(fract(p+ iTime*.25)-.5)*2. - .5; // Snooker.
    //return abs(cos(p*6.283 + iTime))*.5; // Bounce.

}

// Smooth Voronoi. I'm not sure who came up with the original, but I think IQ
// was behind this particular algorithm. It's just like the regular Voronoi
// algorithm, but instead of determining the minimum distance, you accumulate
// values - analogous to adding metaball field values. The result is a nice
// smooth pattern. The "falloff" variable is a smoothing factor of sorts.
//
float smoothVoronoi(vec2 p, float falloff) {

    vec2 ip = floor(p); p -= ip;
	
	float d = 1., res = 0.0;
	
	for(int i = -1; i <= 2; i++) {
		for(int j = -1; j <= 2; j++) {
            
			vec2 b = vec2(i, j);
            
			vec2 v = b - p + hash22(ip + b);
            
			d = max(dot(v,v), 1e-4);
			
			res += 1.0/pow( d, falloff );
		}
	}

	return pow( 1./res, .5/falloff );
}

// 2D function we'll be producing the contours for. 
float func2D(vec2 p){

    
    float d = smoothVoronoi(p*2., 4.)*.66 + smoothVoronoi(p*6., 4.)*.34;
    
    return sqrt(d);
    
}

// Smooth fract function. A bit hacky, but it works. Handy for all kinds of things.
// The final value controls the smoothing, so to speak. Common sense dictates that 
// tighter curves, require more blur, and straighter curves require less. The way 
// you do that is by passing in the function's curve-related value, which in this case
// will be the function value divided by the length of the function's gradient.
//
// IQ's distance estimation example will give you more details:
// Ellipse - Distance Estimation - https://www.shadertoy.com/view/MdfGWn
// There's an accompanying article, which is really insightful, here:
// https://iquilezles.org/articles/distance
float smoothFract(float x, float sf){
 
    x = fract(x); return min(x, x*(1.-x)*sf);
    
}


void main()
{
    vec2 iResolution = vec2(5, 5);
    // Screen coordinates.
	vec2 uv = (fragCoord.xy-iResolution.xy*.5) / iResolution.y;

    // Standard epsilon, used to determine the numerical gradient. 
    vec2 e = vec2(0.001, 0); 

    // The 2D function value. In this case, it's a couple of layers of 2D simplex-like noise.
    // In theory, any function should work.
    float f = func2D(uv); // Range [0, 1]
    
    // Length of the numerical gradient of the function above. Pretty standard. Requires two extra function
    // calls, which isn't too bad.
    float g = length( vec2(f - func2D(uv-e.xy), f - func2D(uv-e.yx)) )/(e.x);
   
    // Dividing a constant by the length of its gradient. Not quite the same, but related to IQ's 
    // distance estimation example: Ellipse - Distance Estimation - https://www.shadertoy.com/view/MdfGWn
    g = 1./max(g, 0.001);
    
    // This is the crux of the shader. Taking a function value and producing some contours. In this case,
    // there are twelve. If you don't care about aliasing, it's as simple as: c = fract(f*12.);
    // If you do, and who wouldn't, you can use the following method. For a quick explanation, refer to the 
    // "smoothFract" function or look up a concetric circle (bullseye) function.
    //
    // For a very good explanation, see IQ's distance estimation example:
    // Ellipse - Distance Estimation - https://www.shadertoy.com/view/MdfGWn
    //
    // There's an accompanying articles, which is really insightful, here:
	// https://iquilezles.org/articles/distance
    //
    float freq = 12.; 
    // Smoothing factor. Hand picked. Ties in with the frequency above. Higher frequencies
    // require a lower value, and vice versa.
    float smoothFactor = iResolution.y*0.0125; 
    
    #ifdef GLOSSY
    float c = smoothFract(f*freq, g*iResolution.y/16.); // Range [0, 1]
    //float c = fract(f*freq); // Aliased version, for comparison.
    #else
    float c = clamp(cos(f*freq*3.14159*2.)*g*smoothFactor, 0., 1.); // Range [0, 1]
    //float c = clamp(cos(f*freq*3.14159*2.)*2., 0., 1.); // Blurry contours, for comparison.
    #endif
    
    
    // Coloring.
    //
    // Convert "c" above to the greyscale and green colors.
    vec3 col = vec3(c);
    vec3 col2 = vec3(c*0.64, c, c*c*0.1);
    
    #ifdef GLOSSY
    col = mix(col, col2, -uv.y + clamp(fract(f*freq*0.5)*2.-1., 0., 1.0));
    #else
    col = mix(col, col2, -uv.y + clamp(cos(f*freq*3.14159)*2., 0., 1.0));
    #endif
    
    // Color in a couple of thecontours above. Not madatory, but it's pretty simple, and an interesting 
    // way to pretty up functions. I use it all the time.
    f = f*freq;
    
    #ifdef GLOSSY
    if(f>8. && f<9.) col *= vec3(1, 0, .1);
    #else
    if(f>8.5 && f<9.5) col *= vec3(1, 0, .1);
    #endif 
   
    
	// Since we have the gradient related value, we may as well use it for something. In this case, we're 
    // adding a bit of highlighting. It's calculated for the contourless noise, so doesn't match up perfectly,
    // but it's good enough. Comment it out to see the texture on its own.  
    #ifdef GLOSSY
    col += g*g*g*vec3(.3, .5, 1)*.25*.25*.25*.1;
    #endif 
    
    
    //col = c * vec3(g*.25); // Just the function and gradient. Has a plastic wrap feel.
	
    // Done.
	fragColor = vec4( sqrt(clamp(col, 0., 1.)), 1.0 );
	
}
"""

compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
compiled_fragment_shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
shader = compileProgram(
    compiled_vertex_shader, 
    compiled_fragment_shader
)
compiled_fragment_shader_2 = compileShader(f_shader_2, GL_FRAGMENT_SHADER)
shader2 = compileProgram(
    compiled_vertex_shader, 
    compiled_fragment_shader_2
)
compiled_fragment_shader_3 = compileShader(f_shader_3, GL_FRAGMENT_SHADER)
shader3 = compileProgram(
    compiled_vertex_shader, 
    compiled_fragment_shader_3
)

glUseProgram(shader)

vertices = []

modelo = Obj('./ave.obj')

for v in modelo.vertices:
    for i in v:
        vertices.append(i)

vertex_data = np.array(vertices, dtype=np.float32)
vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)
vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)

glBufferData(
    GL_ARRAY_BUFFER, # tipo de datos
    vertex_data.nbytes, # tamaño de los datos en bytes
    vertex_data, # puntero a la data
    GL_STATIC_DRAW # tipo de uso de la data
)

glVertexAttribPointer(
    0, 
    3,
    GL_FLOAT,
    GL_FALSE,
    3 * 4,
    ctypes.c_void_p(0)
)

glEnableVertexAttribArray(0)

caras = []
for f in modelo.caras:
    caras.append(f[0][0]-1)
    caras.append(f[1][0]-1)
    caras.append(f[2][0]-1)


caras_data = np.array(caras, dtype=np.int32)
caras_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, caras_buffer_object)

glBufferData(
    GL_ELEMENT_ARRAY_BUFFER, # tipo de datos
    caras_data.nbytes, # tamaño de los datos en bytes
    caras_data, # puntero a la data
    GL_STATIC_DRAW # tipo de uso de la data
)


def calculateMatrix(angle):

    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(0, 0, 0))
    rotate = glm.rotate(i, glm.radians(angle), glm.vec3(0, 1, 0))
    scale = glm.scale(i, glm.vec3(1, 1, 1))

    model = translate * rotate * scale

    view = glm.lookAt(
        glm.vec3(0, 0, 5),
        glm.vec3(0, 0, 0),
        glm.vec3(0, 1, 0)
    )

    projection = glm.perspective(
        glm.radians(45),
        1000 / 1000,
        0.1,
        1000
    )

    glViewport(0, 0, 800, 800)

    matrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader, "matrix"),
        1,
        GL_FALSE,
        glm.value_ptr(matrix)
    )

running = True

glClearColor(0.7, 0.7, 0.5, 1.0)
r = 0 
    
cual = 1
while running:
    glClear(GL_COLOR_BUFFER_BIT)
    

    color1 = random.random()
    color2 = random.random()
    color3 = random.random()

    color = glm.vec3(color1, color2, color3)

    if cual == 1:
        glUniform1f(glGetUniformLocation(shader,'iTime'), pygame.time.get_ticks() / 1000)

        glUniform3fv(
            glGetUniformLocation(shader, "color"),
            1,
            glm.value_ptr(color)
        )
        glUseProgram(shader)
    
    elif cual == 2:
        glUniform1f(glGetUniformLocation(shader2,'iTime'), pygame.time.get_ticks() / 1000)

        glUniform3fv(
            glGetUniformLocation(shader2, "color"),
            1,
            glm.value_ptr(color)
        )
        glUseProgram(shader2)

    elif cual == 3:
        glUniform1f(glGetUniformLocation(shader3,'iTime'), pygame.time.get_ticks() / 1000)

        glUniform3fv(
            glGetUniformLocation(shader3, "color"),
            1,
            glm.value_ptr(color)
        )
        glUseProgram(shader3)

    pygame.time.wait(50)

    calculateMatrix(r)

    #glDrawArrays(GL_TRIANGLES, 0, len(vertex_data))
    glDrawElements(GL_TRIANGLES, len(caras_data), GL_UNSIGNED_INT, None)
    
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if (event.type == pygame.KEYDOWN):
            if event.key == pygame.K_LEFT:
                r -= 1 
            if event.key == pygame.K_RIGHT:
                r += 1
            if event.key == pygame.K_1:
                cual = 1
            if event.key == pygame.K_2:
                cual = 2
            if event.key == pygame.K_3:
                cual = 3 