# Linear Algebra Library For Zig
> zig version: v0.15.1

## Introduction
This is a linear algebra library for Zig aimed towards computer graphics and game development. It provides implementation for common vector, matrix and quaternion operations. At its current stage, it offers support for the following types:
- `Vec2`
- `Vec3`
- `Vec4`
- `Mat2x2`
- `Mat3x3`
- `Mat4x4`
- `Quat`

## Features
- Basic arithmetic operations for vectors and matrices.
- Transposition, inversion, and determinant calculations for matrices.
- Some common vector and matrix transformations.
- Quaternions

## Installation
To add this library to your Zig project using Zig's package manager, append the following to your `build.zig.zon` file:

```zig
.{
    .dependencies = .{
        .zig_matrix = .{
            .url = "https://github.com/AshishBhattarai/zig-matrix/archive/<commit SHA>.tar.gz",
            .hash = "<dependency hash>",
        },
    },
}
```
Replace `<commit SHA>` with the commit SHA of the desired version of the library and `<dependency hash>` with the hash of the dependency.

Now, simply add the following to your `build.zig` file:
```zig
const zig_matrix_dep = b.dependency("zig_matrix", .{
        .target = target,
        .optimize = optimize,
});
root_module.addImport("zig_matrix", zig_matrix_dep.module("zig_matrix"));
```

## Usage
To use this library, simply import it into your Zig project and start utilizing the provided types and functions. Here's a basic example of how to initialize and multiply two `Mat4x4` matrices:

### Matrix Example
```zig
const matrix = @import("zig_matrix");

const a = matrix.mat4x4(
    matrix.vec4(0, 1, 2, 3),
    matrix.vec4(4, 5, 6, 7),
    matrix.vec4(8, 9, 10, 11),
    matrix.vec4(12, 13, 14, 15),
);
const b = matrix.mat4x4(
    matrix.vec4(4, 5, 6, 7),
    matrix.vec4(1, 2, 3, 4),
    matrix.vec4(9, 10, 11, 12),
    matrix.vec4(-1, -2, -3, -4),
);

const result = a.mul(b);
```

### Vector Example
```zig
const matrix = @import("zig_matrix");

const a = matrix.vec2(1, 2);
const b = a.swizzle("xyx");  // b: vec3(1, 2, 1)
const c = a.swizzle("yyxx"); // c: vec4(2, 2, 1, 1)
```
