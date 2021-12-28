# bml

## "Buffer Manipulation Language"

Preprocessor and runtime for basic but programmatic image manipulation

## Building

BML uses Rust, and be built and/or run with `cargo`, which can be installed using [rustup](https://rustup.rs/).

Run 
```bash
cargo run --release <options>
```
while in the directory, or use `cargo build --release` and run the built binary in `./target/release/bml`.

Although building in release mode is not necessary, it is highly recommended for improved performance.

## Usage

When running BML, there are three primary options:

```./bml eval <path/to/script.buf>```

Will simply evaluate the given script and print the `return`-ed result.

```./bml process <path/to/script.buf> <path/to/image.png> <frames> [output_path]```

Will process a given image through a given script. If `<frames>` is specified and greater than 1, a gif will be generated instead of a static image. The output will be written to an automatically determined path, or write to `output_path` instead if valid.

```bml new <script> <width> <height> <frames> <output_path>```

Instead of processing an existing image, `new` will create a new image from scratc using a given script, where `frag` will be set to `[0, 0, 0, 0]` for every pixel. Like `process`, if `<frames>` is greater than 1 the output will be a GIF instead of a PNG.

## Intro

BML itself is a fairly rudimentary C-like language loosely inspired by the likes of shading languages such as GLSL. The premise is that you take in any arbitrary image, and then for each pixel/coordinate return a new one.

As an example, this is a script that will invert every pixel in an image:

```invert.bml
# invert each pixel

return [1 - frag.r, 1 - frag.g, 1 - frag.b, 1]
```

For each pixel, the script will take each component (every image is processed and returned as RGBA) and subtract it from 1.

Similarly, BML also has support for easily generating gifs:

```
# Tom and Jerry-style closing circle

uv = coord / resolution
distance = dist(uv, [0.5, 0.5])
return frag if distance < frame / frame_count else [1; 4]
```

Instead of taking a frame, running each pixel through a script and returning a new image, this script will take an image, a specified number of frames, and for each frame process each pixel in the image with `frame` set to the current frame and `frame_count` set to the number of frames in the gif. 

[Place rocket gif here]

Hence, as `frame / frame_count` approaches 1, more and more of the pixels will have a `uv` too far from the center, and more and more of the gif becomes black.

In summary BML is pretty much a simplified, dynamically-typed GLSL specifically oriented around easily and concisely processing images and gifs.

## Language

Most of the syntax should be familiar to anyone used to a C-family language/GLSL, but there are some idiosyncrasies specific to BML.

### Types

Similar to GLSL, BML has three basic types/primitives:

* **Scalars** or floats

Every number in GLSL is represented as an IEEE 754 32-bit float, or an `f32` in Rust. Unlike GLSL, since every number is a float, `1` and `1.0` are both floats, and can be used for any operations, expressions or statements that expect floats, e.g. vectors or matrices. Note, however, that unlike GLSL syntax like `.5` or `5.` is unsupported and will result in a `ParseError`.

* **Vectors**: BML has support for 2x1 vectors, 3x1 vectors, and a 4x1 vector.

Unlike GLSL, the syntax for vectors in BML are more like arrays in Rust than function calls. You can initialize a vec2, a vec3, or a vec4 with `[1, 2]`, `[1, 2, 3]` or `[1, 2, 3, 4]` respectively. You can also concisely define a vector with a repeated value by putting the value, a `;`, and then the length: e.g., `[1, 1, 1, 1]` becomes `[1; 4]`. This works for any arbitrary expression as well, so the former is equal to `[0.5 * 2 * 7 / 7; 4]`, or `[x; 4]` if `x=1`.

A vector component can be acccessed with the `.` syntax `.x`, `.y`, `.z`, `.w`. So `[1, 2].x` becomes `1`. You can also use the aliases `.rgba` when using colors, so `[0.5, 0.5, 0].r` is `0.5`.

 You can also "swizzle" vectors in any order to get any new vector or float. For example, `[1, 2, 3, 4].xyzw` becomes `[1, 2, 3, 4]`, or `[0.5, 0.5, 0.5].rg` becomes `[0.5, 0.5]`.

 * **Matrices**: BML also suports built-in 2x2, 3x3 and 4x4 matrix primitives.

A matrix is treated as just a list of vectors, hence you can define a matrix by specifying each column using built-in functions. `mat2([1, 0], [0, 1])` becomes 

```
[ 1, 0 ]
[ 0, 1 ]
```

or `mat3([1, 2, 3], [4, 5, 6], [7, 8, 9])` becomes

```
[ 1, 4, 7 ]
[ 2, 5, 8 ]
[ 3, 6, 9 ]
```


Note that matrices are _column-major_ (which could result in some bugs!) so the first argument to `mat2`, `mat3` or `mat4` is the _first column_ and not the first row.

You can index a matrix to get a column with `<matrix>[column]`. So `mat2([1, 0], [0, 1])[0] == [1, 0]`.

You can multiply matrices with vectors to represent transformations. So for instance, to scale a vector `point=[1, 2, 3, 1]` which represents some 3d point, you could scale it by doing `scale(5) * point`, or rotate it with `rotate_x(3.14159 / 2) * point`.

### Variables

Variable declarations are just `<name> = <value>`, where no variable is typed and can be redefined at any time. If a variable is used before it is defined, BML will throw an error:

```
x = 10 * 2 + 1
x = [1, 2]
y = [1, 2].yz
x = y + y

x = z # will throw an error
```

#### Uniforms

For getting the current pixel, coordinate, or gif frame, BML loads some variables into the script namespace:

*vec2 coord*: stores the current pixel coordinate as [x, y]. Note that this is not normalized, so for instance if processign a 100 by 100 pixel image, the `coord` variable would be [0, 0] at the bottom-left and [100, 100] at the top-right.

*vec2 resolution*: stores the width and height of the image as [width, height]

*vec2 frag*: stores current pixel color as [r, g, b, a] where each color is 0-1.

*float frame*: stores current frame if processing a gif. Starts at 0 and ends at `frame_count - 1`

*float frame_count*: stores the specified number of frames if processing a gif.


### Control Flow

For control flow, `BML` has an `if`, `repeat` and `while` statements.

An `if` statement will return the first part if the condition equals 1.0, and return the second part if otherwise:

```
x = 5 if 1 else 1 # x is now equal to 5
y = 2 if 0.5 else 0.25 # y is now 0.25

# == and != will give 1.0 if true, 0.0 if otherwise
# z becomes 1
z = 1 if 5 == 5 else 0.5
```

BML also has block expressions, which can be used for statements, `give`-ing values or both:

```
# x is now 5.5
x = {
    y = 5
    give y + 0.5 # `give` will exit the block and give it a specified value
}


z = 8

# the remainder of z / 2 is 0, so
# z will decrement 
# and y will be set to 0.5
y = {
    z = z - 1 # blocks can both mutate state and be used as expressions
    give 0.5
} if mod(z, 2) == 0 else {
    give 0.2
}
```

The `repeat` statement will just execute a block a specified number of times:

```
i = 0

repeat 100 {
    i = i + 1
}

# i is now 100
```

However the number of times you can repeat `repeat` something is determined at compile-time -- so it must be a literal or a macro.

```
macro iter() 5

# this is run
repeat iter() { }

x = 5 + 2 * coord.x

# this will not
repeat x { }
```

Similarly, with a `while` statement, you can repeat a block a set number of times, but if a given condition becomes false before then BML will exit the loop early.

```
i = 0

while 1 < 50, 100 {
    i = i + 1
}

# i is now 50
```

Like GLSL, since BML only has finite loops that can't be arbitrarily or infinitely long, and only has macros but no runtime functions/recursion, BML is technically not turing-complete and any program must always finish at _some_ point.

### Macros

Instead of functions, for basic code reuse/organization BML offers C-like macros:

```
macro add_one(x) x + 1

y = add_one(10)
z = add_one(5 + {
    give add_one(2) + add_one(y)
})

# expands to
y = 10 + 1
z = 5 + {
    give 2 + 1 + y + 1
} + 1
```

Whenever a macro is invoked, all identifiers in the expansion specified in the parameters in the macro declaration will be replaced respectively with the called arguments. 

If the macro doesn't start with a block, the definition will only extend to the end of the line. However, if the macro does start with a block, every invocation will be replaced with the whole block instead of a single line:

```
macro sum(vec3) {
    give vec3.x + vec3.y + vec3.z
}

macro add_and_multiply(x, y, z) {
    added = x + y + z
    multed = [x, y, z] * [added; 3]
    give sum(multed)
}

x = add_and_multiply(1, 2, 3)

# expands to
x = {
    added = 1 + 2 + 3
    multed = [1, 2, 3] * [added; 3]
    give {
        multed.x + multed.y + multed.z
    }
}
```

### Built-In Functions

For certain functionality like square roots or exponents, BML supports built-in functions baked into the interpreter that are treated like macro calls.

```
# x now equals 3
x = sqrt(9)

# 2^8.5
y = pow(2, 8.5)

# get distance from normalized coordinates to center
uv = coord / resolution
distance = dist(uv, [0.5, 0.5])

# clamp value between 2 and 5
large_value = 1000
clamped = clamp(large_value, 2, 5) # becomes 5
```

Most of the standard shading functions one would expect in GLSL is in BML:

* dist(p0, p1): gets the distance between two floats or vectors
* radians(degrees): converts a float or vector in degrees to be in radians
* degrees(radians): converts a float or vector in radians to be in degrees
* sin(angle): takes sine component-wise of float or vector
* cos(angle): cosine of angle
* tan(angle): tangent of angle
* asin(x): arcsin of x
* acos(x): arccos of x
* atan(x): arctangent of x
* pow(base, exponent): raises base to the power of exponent
* exp(x): exponential function, or e^x
* log(x): natural log of x
* sqrt(x): square root of x
* invsqrt: inverse square root of x
* abs(x): absolute value of x
* sign(x): gets sign of x -- for each component, return 1 if positive, 0 if 0, or -1 if negative
* floor(x): gets the floor of (rounds down) x
* ceil(x): gets the ceiling of (rounds up) x
* fract(x): gets the fractional part of x
* mod(x, y): gets the modulo of x and y
* min(x, y): gets the minimum of x and y
* max(x, y): gets the maximum of x and y
* clamp(x, min, max): "clamps" x between min and max -- returns min if x <= min, max if x >= max, or x if otherwise
* mix(x, y, a): linearly interpolates or "mixes" between x and y according to a (a=0 would return x, a=0.5 would return the value between x and y, and a=1 would be y). If x, y and a are all vectors (assuming they're all the same length) then perform mix component-wise. If x and y are vectors and a is a float, then perform mix component-wise where a is the same float for each component.
* step(edge, x): returns 0 if x is smaller than edge, otherwise returns 1. Will do this component-wise if edge and x are floats/vectors of the same length, or will do it component-wise for x if edge is a scalar and x is a vector.
* length(x): returns the length (or magnitude) of x if x is a vector
* dot(x, y): takes the dot product of x and y
* cross(x, y): takes the cross product of x and y
* norm(x): normalizes x if x is a vector
* mat2(col1, col2): creates a new 2x2 matrix
* mat3(col1, col2, col3): creates a new 3x3 matrix
* mat4(col1, col2, col3, col4): creates a new 4x4 matrix
* rotate_x(radians): returns a 4x4 transformation matrix that rotates a specified number of radians about the x-axis
* rotate_y(radians): returns a 4x4 transformation matrix that rotates about the y-axis
* rotate_z(radians): returns a 4x4 transformation matrix that rotates about the z-axis
* rotate(yaw, pitch, roll): returns a 4x4 transformation matrix that rotataes about the x, y and z axes according to yaw, pitch and roll (all in radians) respectively
* scale(x): returns a 4x4 transformation matrix that scales a point x times
* translate(x, y, z): returns a 4x4 transformation matrix that translates accordingly in the x, y, and z dimensions
* ortho(near, far, left, right, top, bottom): returns a 4x4 transformation matrix for orthographic projection
* lookat(from, to): returns a 4x4 transformation `lookat` matrix
* perspective(fov, near, far)): returns a 4x4 perspective projection matrix

### Sample()

A special function unique to BML is `sample(coordinates)`. Which will return the pixel in the original image (assuming that a pre-existing iamge is being processed) according to any arbitrary normalized x and y coordinates from 0-1. E.g., `sample([0, 0])` returns the pixel at the bottom-left of the image, and `sample([1, 1])` returns the pixel at the top-right. 

Although the `frag` variable already gives you the color of the current pixel, being able to arbitrarily sample from the image or around the current pixel can be useful for things such as certain distortions.

#### `return`-ing

A BML program consists of a series of statements which at some point must return a new RGBA color represented as a 4x1 vector `[R, G, B, A]`, where each component is from 0 to 1, using the `return` statement. Going back to the closing circle example:

```
# Tom and Jerry-style closing circle

uv = coord / resolution
distance = dist(uv, [0.5, 0.5])
return frag if distance < frame / frame_count else [1; 4]
```

How the script returns the new pixel color is through that final `return` statement. Note that this is short-circuiting:

```
# Tom and Jerry-style closing circle

uv = coord / resolution
distance = dist(uv, [0.5, 0.5])

{
    # if condition is true, then program would end here
    return frag
} if distance < frame / frame_count>

# does not run if condition is true
return [1; 4]
```