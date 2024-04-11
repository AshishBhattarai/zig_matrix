const GenericVector = @import("vector.zig").GenericVector;
const GenericMatrix = @import("matrix.zig").GenericMatrix;

pub const Vec2 = GenericVector(2, f32);
pub const Vec3 = GenericVector(3, f32);
pub const Vec4 = GenericVector(4, f32);
pub const Vec2d = GenericVector(2, f64);
pub const Vec3d = GenericVector(3, f64);
pub const Vec4d = GenericVector(4, f64);
pub const Vec2h = GenericVector(2, f16);
pub const Vec3h = GenericVector(3, f16);
pub const Vec4h = GenericVector(4, f16);

pub const Mat2x2 = GenericMatrix(2, 2, f32);
pub const Mat3x3 = GenericMatrix(3, 3, f32);
pub const Mat4x4 = GenericMatrix(4, 4, f32);
pub const Mat2x2d = GenericMatrix(2, 2, f64);
pub const Mat3x3d = GenericMatrix(3, 3, f64);
pub const Mat4x4d = GenericMatrix(4, 4, f64);
pub const Mat2x2h = GenericMatrix(2, 2, f16);
pub const Mat3x3h = GenericMatrix(3, 3, f16);
pub const Mat4x4h = GenericMatrix(4, 4, f16);

pub const vec2 = Vec2.init;
pub const vec3 = Vec3.init;
pub const vec4 = Vec4.init;
pub const vec2d = Vec2d.init;
pub const vec3d = Vec3d.init;
pub const vec4d = Vec4d.init;
pub const vec2h = Vec2h.init;
pub const vec3h = Vec3h.init;
pub const vec4h = Vec4h.init;

pub const mat2x2 = Mat2x2.init;
pub const mat3x3 = Mat3x3.init;
pub const mat4x4 = Mat4x4.init;
pub const mat2x2d = Mat2x2d.init;
pub const mat3x3d = Mat3x3d.init;
pub const mat4x4d = Mat4x4d.init;
pub const mat2x2h = Mat2x2h.init;
pub const mat3x3h = Mat3x3h.init;
pub const mat4x4h = Mat4x4h.init;

test {
    @import("std").testing.refAllDeclsRecursive(@This());
}
