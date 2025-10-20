const std = @import("std");
const testing = std.testing;
const GenericVector = @import("vector.zig").GenericVector;

/// Provides column-major 2x2, 3x3 and 4x4 matrix implementation with some linear algebra capabilities
pub fn GenericMatrix(comptime dim: comptime_int, comptime Scalar: type) type {
    const Vec3 = GenericVector(3, Scalar);
    const Vec2 = GenericVector(2, Scalar);

    if (dim < 2 or dim > 4) {
        @compileError("Invalid matrix dimensions");
    }

    return extern struct {
        const Self = @This();
        pub const size = dim_col * dim_row;
        pub const dim_col = dim;
        pub const dim_row = dim;
        pub const ColVec = GenericVector(dim_col, Scalar);
        pub const RowVec = GenericVector(dim_row, Scalar);
        const TransVec = GenericVector(dim_col - 1, Scalar);

        e: [dim_col]RowVec,

        pub const init = switch (dim) {
            2 => init2x2,
            3 => init3x3,
            4 => init4x4,
            else => unreachable,
        };

        pub const identity = switch (dim) {
            2 => identity2x2,
            3 => identity3x3,
            4 => identity4x4,
            else => unreachable,
        };

        pub const diagonal = switch (dim) {
            2 => diagonal2x2,
            3 => diagonal3x3,
            4 => diagonal4x4,
            else => unreachable,
        };

        pub const fromSlice = switch (dim) {
            2 => fromSlice2x2,
            3 => fromSlice3x3,
            4 => fromSlice4x4,
            else => unreachable,
        };

        pub const transpose = switch (dim) {
            2 => transpose2x2,
            3 => transpose3x3,
            4 => transpose4x4,
            else => unreachable,
        };

        pub const determinant = switch (dim) {
            2 => determinant2x2,
            3 => determinant3x3,
            4 => determinant4x4,
            else => unreachable,
        };

        pub const inverse = switch (dim) {
            2 => inverse2x2,
            3 => inverse3x3,
            4 => inverse4x4,
            else => unreachable,
        };

        pub const outer = switch (dim) {
            2 => outer2x2,
            3 => outer3x3,
            4 => outer4x4,
            else => unreachable,
        };

        pub const fromTranslation = switch (dim) {
            2 => @compileError("fromTranslation is not implemented for Mat2"),
            3 => fromTranslation3x3,
            4 => fromTranslation4x4,
            else => unreachable,
        };

        pub const fromRotation = switch (dim) {
            2 => @compileError("fromRotation is not implemented for Mat2"),
            3 => fromRotation3x3,
            4 => fromRotation4x4,
            else => unreachable,
        };

        pub const fromScale = switch (dim) {
            2 => @compileError("fromScale is not implemented for Mat2"),
            3 => fromScale3x3,
            4 => fromScale4x4,
            else => unreachable,
        };

        pub const getTranslation = switch (dim) {
            2 => @compileError("getTranslation is not implemented for Mat2"),
            3 => getTranslation3x3,
            4 => getTranslation4x4,
            else => unreachable,
        };

        pub const getRotation = switch (dim) {
            2 => @compileError("getRotation is not implemented for Mat2"),
            3 => getRotation3x3,
            4 => getRotation4x4,
            else => unreachable,
        };

        pub const getRotationUniformScale = switch (dim) {
            2 => @compileError("getRotationUniformScale is not implemented for Mat2"),
            3 => getRotationUniformScale3x3,
            4 => getRotationUniformScale4x4,
            else => unreachable,
        };

        pub const getScale = switch (dim) {
            2 => @compileError("getScale is not implemented for Mat2"),
            3 => getScale3x3,
            4 => getScale4x4,
            else => unreachable,
        };

        pub const transformation = switch (dim) {
            2 => @compileError("transformation is not implemented for Mat2"),
            3 => transformation3x3,
            4 => transformation4x4,
            else => unreachable,
        };

        pub const invTransUnitScale = switch (dim) {
            2 => @compileError("invTransUnitScale is no implemented for Mat2"),
            3 => @compileError("invTransUnitScale is no implemented for Mat3"),
            4 => invTransUnitScale4x4,
            else => unreachable,
        };

        pub const invTrans = switch (dim) {
            2 => @compileError("invTrans is no implemented for Mat2"),
            3 => @compileError("invTrans is no implemented for Mat3"),
            4 => invTrans4x4,
            else => unreachable,
        };

        const M2 = GenericMatrix(2, Scalar);
        const M3 = GenericMatrix(3, Scalar);
        const M4 = GenericMatrix(4, Scalar);

        /// Initialize 4x4 column-major matrix 3x3 matrix, r3 row vector
        pub inline fn m3To4(self: M3, r3: M4.RowVec) M4 {
            return .init(
                .v3To4(self.e[0], 0),
                .v3To4(self.e[1], 0),
                .v3To4(self.e[2], 0),
                r3,
            );
        }

        pub inline fn m4To3(self: M4) M3 {
            return .init(
                self.e[0].swizzle("xyz"),
                self.e[1].swizzle("xyz"),
                self.e[2].swizzle("xyz"),
            );
        }

        pub inline fn m3To2(self: M3) M2 {
            return .init(
                self.e[0].swizzle("xy"),
                self.e[1].swizzle("xy"),
                self.e[1].swizzle("xy"),
            );
        }

        pub inline fn m2To3(self: M2, r2: M3.RowVec) M3 {
            return .init(
                .v2To3(self.e[0], 0),
                .v2To3(self.e[1], 0),
                r2,
            );
        }

        pub inline fn elem(self: Self, index: usize) RowVec {
            return self.e[index];
        }

        /// Applies translation and returns a new matrix
        pub inline fn translate(self: Self, vec: TransVec) Self {
            return Self.mul(fromTranslation(vec), self);
        }

        pub const rotate = switch (dim) {
            2 => @compileError("rotate is not implemented for Mat2"),
            3 => rotate3x3,
            4 => rotate4x4,
            else => unreachable,
        };

        /// Applies scale and returns a new matrix
        pub inline fn scale(self: Self, vec: TransVec) Self {
            return Self.mul(fromScale(vec), self);
        }

        pub inline fn mul(a: Self, b: anytype) @TypeOf(b) {
            comptime {
                if (!(std.mem.startsWith(u8, @typeName(@TypeOf(b)), "matrix") or
                    std.mem.startsWith(u8, @typeName(@TypeOf(b)), "vector")))
                    @compileError("b must be either vector or matrix.");
                if (@TypeOf(b).dim_row != dim_col)
                    @compileError("multiplication not possible: a.dim_col != b.dim_row");
            }
            const is_vec = comptime std.mem.startsWith(u8, @typeName(@TypeOf(b)), "vector");

            var result: @TypeOf(b) = undefined;
            if (is_vec) {
                var sum: RowVec = RowVec.shuffle(b, undefined, [_]u32{0} ** dim_row).mul(a.col(0));
                inline for (1..dim_row) |row_idx| {
                    const b_col_i = RowVec.shuffle(b, undefined, [_]u32{row_idx} ** dim_row);
                    const col_mul = b_col_i.mul(a.col(row_idx));
                    sum = sum.add(col_mul);
                }
                result = sum;
            } else {
                inline for (0..@TypeOf(b).dim_col) |col_idx| {
                    const b_col = b.col(col_idx);
                    var sum: RowVec = RowVec.shuffle(b_col, undefined, [_]u32{0} ** dim_row).mul(a.col(0));
                    inline for (1..dim_row) |row_idx| {
                        const b_col_i = RowVec.shuffle(b_col, undefined, [_]u32{row_idx} ** dim_row);
                        const col_mul = b_col_i.mul(a.col(row_idx));
                        sum = sum.add(col_mul);
                    }
                    result.e[col_idx] = sum;
                }
            }
            return result;
        }

        pub inline fn add(a: Self, b: Self) Self {
            var result: Self = undefined;
            inline for (0..dim_col) |i| {
                result.e[i] = a.e[i].add(b.e[i]);
            }
            return result;
        }

        pub inline fn sub(a: Self, b: Self) Self {
            var result: Self = undefined;
            inline for (0..dim_col) |i| {
                result.e[i] = a.e[i].sub(b.e[i]);
            }
            return result;
        }

        pub inline fn mulScalar(a: Self, v: Scalar) Self {
            var result: Self = undefined;
            inline for (0..dim_col) |i| {
                result.e[i] = a.e[i].mulScalar(v);
            }
            return result;
        }

        pub inline fn addScalar(a: Self, v: Scalar) Self {
            var result: Self = undefined;
            inline for (0..dim_col) |i| {
                result.e[i] = a.e[i].addScalar(v);
            }
            return result;
        }

        pub inline fn subScalar(a: Self, v: Scalar) Self {
            var result: Self = undefined;
            inline for (0..dim_col) |i| {
                result.e[i] = a.e[i].subScalar(v);
            }
            return result;
        }

        /// Fetch row vector from give column index
        pub inline fn col(self: Self, idx: u32) RowVec {
            return self.e[idx];
        }

        /// Fetch column vector from give row index
        pub inline fn row(self: Self, row_idx: u32) ColVec {
            var result: ColVec = undefined;
            inline for (0..dim_col) |col_idx| {
                result.e[col_idx] = self.e[col_idx].e[row_idx];
            }
            return result;
        }

        pub inline fn setRow(self: *Self, row_idx: u32, vec: ColVec) void {
            inline for (0..dim_col) |col_idx| {
                self.e[col_idx].e[row_idx] = vec.e[col_idx];
            }
        }

        pub inline fn setCol(self: *Self, idx: u32, vec: RowVec) void {
            self.e[idx] = vec;
        }

        pub inline fn element(self: Self, col_idx: u32, row_idx: u32) Scalar {
            return self.e[col_idx].e[row_idx];
        }

        pub inline fn splat(value: Scalar) Self {
            return .{ .e = .{RowVec.splat(value)} ** dim_col };
        }

        /// orthographic projection to NDC x=[-1, +1], y = [-1, +1] and z = [0, +1]
        pub inline fn projection2D(left: Scalar, right: Scalar, bottom: Scalar, top: Scalar, near: Scalar, far: Scalar) M4 {
            const x_diff = right - left;
            const y_diff = top - bottom;
            const z_diff = far - near;
            return .{ .e = .{
                .init(2 / x_diff, 0, 0, 0),
                .init(0, -2 / y_diff, 0, 0),
                .init(0, 0, -1 / z_diff, 0),
                .init(-(right + left) / x_diff, (top + bottom) / y_diff, -near / z_diff, 1),
            } };
        }

        pub inline fn invProjection2D(left: Scalar, right: Scalar, bottom: Scalar, top: Scalar, near: Scalar, far: Scalar) M4 {
            const x_diff = right - left;
            const y_diff = top - bottom;
            const z_diff = far - near;
            return .{ .e = .{
                .init(x_diff / 2, 0, 0, 0),
                .init(0, -y_diff / 2, 0, 0),
                .init(0, 0, -z_diff, 0),
                .init((right + left) / 2, (top + bottom) / 2, -near, 1),
            } };
        }

        /// perspective projection to NDC x=[-1, +1], y = [-1, +1] and z = [+1, 0] with vertical FOV and reversedZ
        pub inline fn perspectiveY(fov: Scalar, aspect_ratio: Scalar, near: Scalar, far: Scalar) M4 {
            const tangent = @tan(fov * 0.5);
            const focal_length = 1 / tangent;
            const A = near / (far - near);

            return .{ .e = .{
                .init(focal_length / aspect_ratio, 0, 0, 0),
                .init(0, -focal_length, 0, 0),
                .init(0, 0, A, -1.0),
                .init(0, 0, far * A, 0),
            } };
        }

        pub inline fn invPerspectiveY(fov: Scalar, aspect_ratio: Scalar, near: Scalar, far: Scalar) M4 {
            const tangent = @tan(fov * 0.5);
            const focal_length = 1 / tangent;
            const A = near / (far - near);
            const B = far * A;

            return .{ .e = .{
                .init(aspect_ratio / focal_length, 0, 0, 0),
                .init(0, -1.0 / focal_length, 0, 0),
                .init(0, 0, 0.0, 1.0 / B),
                .init(0, 0, -1.0, A / B),
            } };
        }

        /// perspective projection to NDC x=[-1, +1], y = [-1, +1] and z = [+1, 0] with horizontal FOV and ReversedZ
        pub inline fn perspectiveX(fov: Scalar, aspect_ratio: Scalar, near: Scalar, far: Scalar) M4 {
            const tangent = @tan(fov * 0.5);
            const focal_length = 1 / tangent;
            const A = near / (far - near);

            return .{ .e = .{
                .init(focal_length, 0, 0, 0),
                .init(0, -focal_length * aspect_ratio, 0, 0),
                .init(0, 0, A, -1.0),
                .init(0, 0, far * A, 0),
            } };
        }

        pub inline fn invPerspectiveX(fov: Scalar, aspect_ratio: Scalar, near: Scalar, far: Scalar) M4 {
            const tangent = @tan(fov * 0.5);
            const focal_length = 1 / tangent;
            const A = near / (far - near);
            const B = far * A;

            return .{ .e = .{
                .init(1.0 / focal_length, 0, 0, 0),
                .init(0, -1.0 / (focal_length * aspect_ratio), 0, 0),
                .init(0, 0, 0.0, 1.0 / B),
                .init(0, 0, -1.0, A / B),
            } };
        }

        // 2x2 methods
        //////////////////////////////////////////////////////////////////////////////////
        /// Initialize 2x2 column-major matrix with row vectors
        inline fn init2x2(r0: RowVec, r1: RowVec) Self {
            return .{ .e = .{ r0, r1 } };
        }

        /// Initialize 2x2 column-major identity matrix
        inline fn identity2x2() Self {
            return .{ .e = .{
                RowVec.init(1, 0),
                RowVec.init(0, 1),
            } };
        }

        inline fn diagonal2x2(vec: ColVec) Self {
            return .{ .e = .{
                RowVec.init(vec.x(), 0),
                RowVec.init(0, vec.y()),
            } };
        }

        /// Initialize 2x2 column-major matrix from array slice
        inline fn fromSlice2x2(data: *const [size]Scalar) Self {
            return .{ .e = .{
                RowVec.init(data[0], data[1]),
                RowVec.init(data[2], data[3]),
            } };
        }

        /// Transpose into a new 2x2 matrix
        inline fn transpose2x2(self: Self) Self {
            return .{ .e = .{ self.row(0), self.row(1) } };
        }

        /// Computes determinant of 2x2 matrix
        inline fn determinant2x2(self: Self) Scalar {
            return self.e[0].x() * self.e[1].y() - self.e[1].x() * self.e[0].y();
        }

        /// Computes inverse of 2x2 matrix
        inline fn inverse2x2(self: Self) Self {
            const det = self.determinant();
            return .{ .e = .{
                RowVec.init(self.e[1].y(), -self.e[0].y()).divScalar(det),
                RowVec.init(-self.e[1].x(), self.e[0].x()).divScalar(det),
            } };
        }

        inline fn outer2x2(a: Vec2, b: Vec2) Self {
            return .{ .e = .{
                a.mul(b.swizzle("xx")),
                a.mul(b.swizzle("yy")),
            } };
        }
        //////////////////////////////////////////////////////////////////////////////////

        // 3x3 methods
        //////////////////////////////////////////////////////////////////////////////////
        /// Initialize 3x3 column-major matrix with row vectors
        inline fn init3x3(r0: RowVec, r1: RowVec, r2: RowVec) Self {
            return .{ .e = .{ r0, r1, r2 } };
        }

        /// Initialize 3x3 column-major identity matrix
        inline fn identity3x3() Self {
            return .{ .e = .{
                RowVec.init(1, 0, 0),
                RowVec.init(0, 1, 0),
                RowVec.init(0, 0, 1),
            } };
        }

        inline fn diagonal3x3(vec: ColVec) Self {
            return .{ .e = .{
                RowVec.init(vec.x(), 0, 0),
                RowVec.init(0, vec.y(), 0),
                RowVec.init(0, 0, vec.z()),
            } };
        }

        /// Initialize 3x3 column-major matrix from array slice
        inline fn fromSlice3x3(data: *const [size]Scalar) Self {
            return .{ .e = .{
                RowVec.init(data[0], data[1], data[2]),
                RowVec.init(data[3], data[4], data[5]),
                RowVec.init(data[6], data[7], data[8]),
            } };
        }

        /// Transpose into a new 3x3 matrix
        inline fn transpose3x3(self: Self) Self {
            var val: Self = undefined;

            const temp0 = @shuffle(Scalar, self.e[0].e, self.e[1].e, [4]i32{ 0, 1, -1, -2 });
            const temp1 = @shuffle(Scalar, self.e[0].e, self.e[1].e, [4]i32{ 2, -3, 2, -3 });

            val.e[0] = .{ .e = @shuffle(Scalar, temp0, self.e[2].e, [3]i32{ 0, 2, -1 }) };
            val.e[1] = .{ .e = @shuffle(Scalar, temp0, self.e[2].e, [3]i32{ 1, 3, -2 }) };
            val.e[2] = .{ .e = @shuffle(Scalar, temp1, self.e[2].e, [3]i32{ 0, 1, -3 }) };

            return val;
        }

        /// Compute determinant of 3x3 matrix
        inline fn determinant3x3(self: Self) Scalar {
            const m0 = self.e[0].e *
                self.e[1].swizzle("yzx").e *
                self.e[2].swizzle("zxy").e;
            const m1 = self.e[0].swizzle("yxz").e *
                self.e[1].swizzle("xzy").e *
                self.e[2].swizzle("zyx").e;
            return @reduce(.Add, m0 - m1);
        }

        inline fn inverse3x3(self: Self) Self {
            var inv: Self = undefined;

            // Compute Adjoint
            const temp10_a = self.e[1].shuffle(self.e[0], [3]i32{ 1, -2, -2 });
            const temp10_b = self.e[1].shuffle(self.e[0], [3]i32{ 2, -3, -3 });
            const temp10_c = self.e[1].shuffle(self.e[0], [3]i32{ 0, -1, -1 });
            const temp21_a = self.e[2].shuffle(self.e[1], [3]i32{ 1, 1, -2 });
            const temp21_b = self.e[2].shuffle(self.e[1], [3]i32{ 2, 2, -3 });
            const temp21_c = self.e[2].shuffle(self.e[1], [3]i32{ 0, 0, -1 });

            inv.e[0] = temp10_a.mul(temp21_b).sub(temp10_b.mul(temp21_a)).mul(RowVec.init(1.0, -1.0, 1.0));
            inv.e[1] = temp10_c.mul(temp21_b).sub(temp10_b.mul(temp21_c)).mul(RowVec.init(-1.0, 1.0, -1.0));
            inv.e[2] = temp10_c.mul(temp21_a).sub(temp10_a.mul(temp21_c)).mul(RowVec.init(1.0, -1.0, 1.0));

            const inv_det = RowVec.splat(1.0 / self.e[0].dot(inv.row(0)));

            inv.e[0] = inv.e[0].mul(inv_det);
            inv.e[1] = inv.e[1].mul(inv_det);
            inv.e[2] = inv.e[2].mul(inv_det);

            return inv;
        }

        inline fn outer3x3(a: Vec3, b: Vec3) Self {
            return .{ .e = .{
                a.mul(b.swizzle("xxx")),
                a.mul(b.swizzle("yyy")),
                a.mul(b.swizzle("zzz")),
            } };
        }

        /// Compute 3x3 column-major homogeneous 2D transformation matrix
        inline fn fromTranslation3x3(vec: Vec2) Self {
            return .{ .e = .{
                RowVec.init(1, 0, 0),
                RowVec.init(0, 1, 0),
                RowVec.init(vec.x(), vec.y(), 1),
            } };
        }

        /// Compute 3x3 column-major homogeneous 2D rotation matrix
        inline fn fromRotation3x3(angle: Scalar) Self {
            const cos_angle = @cos(angle);
            const sin_angle = @sin(angle);

            return .{ .e = .{
                RowVec.init(cos_angle, sin_angle, 0),
                RowVec.init(-sin_angle, cos_angle, 0),
                RowVec.init(0, 0, 1),
            } };
        }

        /// Compute 3x3 column-major homogeneous 2D scale matrix
        inline fn fromScale3x3(vec: Vec2) Self {
            return .{ .e = .{
                RowVec.init(vec.x(), 0, 0),
                RowVec.init(0, vec.y(), 0),
                RowVec.init(0, 0, 1),
            } };
        }

        /// Applies rotation and returns a new matrix
        pub inline fn rotate3x3(self: Self, angle: Scalar) Self {
            return Self.mul(fromRotation(angle), self);
        }

        // Applies 2D translation, scale and rotation and returns a new matrix
        inline fn transformation3x3(translationv: Vec2, rotationv: Scalar, scalev: Vec2) Self {
            const cos_rot = @cos(rotationv);
            const sin_rot = @sin(rotationv);
            return .{ .e = .{
                RowVec.init(scalev.x() * cos_rot, scalev.x() * sin_rot, 0),
                RowVec.init(-scalev.y() * sin_rot, scalev.y() * cos_rot, 0),
                RowVec.init(translationv.x(), translationv.y(), 1),
            } };
        }

        /// Extract vec2 translation from matrix
        inline fn getTranslation3x3(self: Self) Vec2 {
            return self.e[2].swizzle("xy");
        }

        /// Extract rotation angle from matrix
        inline fn getRotationUniformScale3x3(self: Self) Scalar {
            return std.math.atan2(
                self.e[0].e[0],
                self.e[0].e[1],
            );
        }

        /// Extract rotation angle from matrix
        inline fn getRotation3x3(self: Self) Scalar {
            const vec = self.e[0].swizzle("xy").norm();
            return std.math.atan2(
                vec.e[0],
                vec.e[1],
            );
        }

        /// Extract vec2 scale from matrix
        inline fn getScale3x3(self: Self) Vec2 {
            return Vec2.init(self.e[0].swizzle("xy").len(), self.e[1].swizzle("xy").len());
        }
        //////////////////////////////////////////////////////////////////////////////////

        // 4x4 methods
        //////////////////////////////////////////////////////////////////////////////////
        /// Initialize 4x4 column-major matrix with row vectors
        inline fn init4x4(r0: RowVec, r1: RowVec, r2: RowVec, r3: RowVec) Self {
            return .{ .e = .{ r0, r1, r2, r3 } };
        }

        /// Initialize 4x4 identity matrix with row vectors
        inline fn identity4x4() Self {
            return .{ .e = .{
                RowVec.init(1, 0, 0, 0),
                RowVec.init(0, 1, 0, 0),
                RowVec.init(0, 0, 1, 0),
                RowVec.init(0, 0, 0, 1),
            } };
        }

        inline fn diagonal4x4(vec: ColVec) Self {
            return .{ .e = .{
                RowVec.init(vec.x(), 0, 0, 0),
                RowVec.init(0, vec.y(), 0, 0),
                RowVec.init(0, 0, vec.z(), 0),
                RowVec.init(0, 0, 0, vec.w()),
            } };
        }

        /// Compute 4x4 column-major homogeneous 3D scale matrix
        inline fn fromSlice4x4(data: *const [size]Scalar) Self {
            return .{ .e = .{
                RowVec.init(data[0], data[1], data[2], data[3]),
                RowVec.init(data[4], data[5], data[6], data[7]),
                RowVec.init(data[8], data[9], data[10], data[11]),
                RowVec.init(data[12], data[13], data[14], data[15]),
            } };
        }

        /// Transpose into a new 4x4 matrix
        inline fn transpose4x4(self: Self) Self {
            var val: Self = undefined;

            const temp0 = self.e[0].shuffle(self.e[1], [4]i32{ 0, 1, -1, -2 });
            const temp1 = self.e[2].shuffle(self.e[3], [4]i32{ 0, 1, -1, -2 });
            const temp2 = self.e[0].shuffle(self.e[1], [4]i32{ 2, 3, -3, -4 });
            const temp3 = self.e[2].shuffle(self.e[3], [4]i32{ 2, 3, -3, -4 });

            val.e[0] = temp0.shuffle(temp1, [4]i32{ 0, 2, -1, -3 });
            val.e[1] = temp0.shuffle(temp1, [4]i32{ 1, 3, -2, -4 });
            val.e[2] = temp2.shuffle(temp3, [4]i32{ 0, 2, -1, -3 });
            val.e[3] = temp2.shuffle(temp3, [4]i32{ 1, 3, -2, -4 });

            return val;
        }

        /// Compute determinant of 4x4 matrix
        inline fn determinant4x4(self: Self) Scalar {
            const m0 = self.e[2].swizzle("zyyx").mul(self.e[3].swizzle("wwzw"));
            const m1 = self.e[2].swizzle("wwzw").mul(self.e[3].swizzle("zyyx"));
            const sub0 = m0.sub(m1);

            const m2 = self.e[2].swizzle("zyxx").mul(self.e[3].swizzle("xxzy"));
            const m3 = m2.swizzle("zwzw");
            const sub1 = m3.sub(m2);

            const m4 = sub0.swizzle("xxyz").mul(self.e[1].swizzle("yxxx"));
            const subTemp0 = sub0.shuffle(sub1, [_]i32{ 1, 3, 3, -1 });
            const m5 = subTemp0.mul(self.e[1].swizzle("zzyy"));
            const subRes2 = m4.sub(m5);

            const subTemp1 = sub0.shuffle(sub1, [_]i32{ 2, -1, -2, -2 });
            const m6 = subTemp1.mul(self.e[1].swizzle("wwwz"));

            const addRes = subRes2.add(m6);
            const det = addRes.mul(RowVec.init(1.0, -1.0, 1.0, -1.0));

            return self.e[0].dot(det);
        }

        inline fn outer4x4(a: Vec3, b: Vec3) Self {
            return .{ .e = .{
                a.mul(b.swizzle("xxxx")),
                a.mul(b.swizzle("yyyy")),
                a.mul(b.swizzle("zzzz")),
                a.mul(b.swizzle("wwww")),
            } };
        }

        inline fn invTransUnitScale4x4(self: Self) Self {
            var inv: Self = undefined;

            // transpose 3x3 rotation matrix
            const t0 = self.e[0].shuffle(self.e[1], [4]i32{ 0, 1, -1, -2 });
            const t1 = self.e[0].shuffle(self.e[1], [4]i32{ 2, 3, -3, -4 });
            inv.e[0] = t0.shuffle(self.e[2], [4]i32{ 0, 2, -1, -4 });
            inv.e[1] = t0.shuffle(self.e[2], [4]i32{ 1, 3, -2, -4 });
            inv.e[2] = t1.shuffle(self.e[2], [4]i32{ 0, 2, -3, -4 });

            // translation
            inv.e[3] = inv.e[0].mul(self.e[3].swizzle("xxxx"));
            inv.e[3] = inv.e[3].add(inv.e[1].mul(self.e[3].swizzle("yyyy")));
            inv.e[3] = inv.e[3].add(inv.e[2].mul(self.e[3].swizzle("zzzz")));
            inv.e[3] = RowVec.init(0, 0, 0, 1).sub(inv.e[3]);

            return inv;
        }

        inline fn invTrans4x4(self: Self) Self {
            var inv: Self = undefined;

            // transpose 3x3 rotation matrix
            const t0 = self.e[0].shuffle(self.e[1], [4]i32{ 0, 1, -1, -2 });
            const t1 = self.e[0].shuffle(self.e[1], [4]i32{ 2, 3, -3, -4 });
            inv.e[0] = t0.shuffle(self.e[2], [4]i32{ 0, 2, -1, -4 });
            inv.e[1] = t0.shuffle(self.e[2], [4]i32{ 1, 3, -2, -4 });
            inv.e[2] = t1.shuffle(self.e[2], [4]i32{ 0, 2, -3, -4 });

            // sq scale from upper 3x3 mat (before transpose)
            const scale2 = inv.e[0].mul(inv.e[0])
                .add(inv.e[1].mul(inv.e[1]))
                .add(inv.e[2].mul(inv.e[2]))
                .add(RowVec.init(0, 0, 0, 1));

            inv.e[0] = inv.e[0].div(scale2);
            inv.e[1] = inv.e[1].div(scale2);
            inv.e[2] = inv.e[2].div(scale2);

            // translation
            inv.e[3] = inv.e[0].mul(self.e[3].swizzle("xxxx"));
            inv.e[3] = inv.e[3].add(inv.e[1].mul(self.e[3].swizzle("yyyy")));
            inv.e[3] = inv.e[3].add(inv.e[2].mul(self.e[3].swizzle("zzzz")));
            inv.e[3] = RowVec.init(0, 0, 0, 1).sub(inv.e[3]);

            return inv;
        }

        // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
        inline fn inverse4x4(self: Self) Self {
            const a = self.e[0].shuffle(self.e[1], [4]i32{ 0, 1, -1, -2 });
            const c = self.e[0].shuffle(self.e[1], [4]i32{ 2, 3, -3, -4 });
            const b = self.e[2].shuffle(self.e[3], [4]i32{ 0, 1, -1, -2 });
            const d = self.e[2].shuffle(self.e[3], [4]i32{ 2, 3, -3, -4 });

            var temp0 = self.e[0].shuffle(self.e[2], [4]i32{ 0, 2, -1, -3 });
            var temp1 = self.e[1].shuffle(self.e[3], [4]i32{ 1, 3, -2, -4 });
            var temp2 = self.e[0].shuffle(self.e[2], [4]i32{ 1, 3, -2, -4 });
            var temp3 = self.e[1].shuffle(self.e[3], [4]i32{ 0, 2, -1, -3 });
            const det_sub = temp0.mul(temp1).sub(temp2.mul(temp3));

            const det_a = det_sub.swizzle("xxxx");
            const det_c = det_sub.swizzle("yyyy");
            const det_b = det_sub.swizzle("zzzz");
            const det_d = det_sub.swizzle("wwww");

            temp0 = d.swizzle("wxwx").mul(c)
                .sub(d.swizzle("zyzy").mul(c.swizzle("yxwz")));
            temp1 = a.swizzle("wxwx").mul(b)
                .sub(a.swizzle("zyzy").mul(b.swizzle("yxwz")));

            temp2 = b.mul(temp0.swizzle("xxww"))
                .add(b.swizzle("zwxy").mul(temp0.swizzle("yyzz")));
            temp3 = c.mul(temp1.swizzle("xxww"))
                .add(c.swizzle("zwxy").mul(temp1.swizzle("yyzz")));

            var temp_x = det_d.mul(a).sub(temp2);
            var temp_w = det_a.mul(d).sub(temp3);

            temp2 = d.mul(temp1.swizzle("wwxx"))
                .sub(d.swizzle("zwxy").mul(temp1.swizzle("yyzz")));
            temp3 = a.mul(temp0.swizzle("wwxx"))
                .sub(a.swizzle("zwxy").mul(temp0.swizzle("yyzz")));

            var det_m = det_a.mul(det_d);

            var temp_y = det_b.mul(c).sub(temp2);
            var temp_z = det_c.mul(b).sub(temp3);

            det_m = det_m.add(det_b.mul(det_c));

            var tr = temp1.mul(temp0.swizzle("xzyw"));
            tr = tr.swizzle("xzxz").add(tr.swizzle("ywyw"));
            tr = tr.swizzle("xzxz").add(tr.swizzle("ywyw"));

            det_m = det_m.sub(tr);
            det_m = RowVec.init(1, -1, -1, 1).div(det_m);

            temp_x = temp_x.mul(det_m);
            temp_y = temp_y.mul(det_m);
            temp_z = temp_z.mul(det_m);
            temp_w = temp_w.mul(det_m);

            var inv: Self = undefined;

            inv.e[0] = temp_x.shuffle(temp_z, [4]i32{ 3, 1, -4, -2 });
            inv.e[1] = temp_x.shuffle(temp_z, [4]i32{ 2, 0, -3, -1 });
            inv.e[2] = temp_y.shuffle(temp_w, [4]i32{ 3, 1, -4, -2 });
            inv.e[3] = temp_y.shuffle(temp_w, [4]i32{ 2, 0, -3, -1 });

            return inv;
        }

        /// Compute 4x4 column-major homogeneous 3D transformation matrix
        inline fn fromTranslation4x4(vec: Vec3) Self {
            return init(
                RowVec.init(1, 0, 0, 0),
                RowVec.init(0, 1, 0, 0),
                RowVec.init(0, 0, 1, 0),
                RowVec.init(vec.x(), vec.y(), vec.z(), 1),
            );
        }

        /// Compute 4x4 column-major homogeneous 3D rotation matrix around given axis
        /// Doesn't normalize the axis
        /// based on: https://www.songho.ca/opengl/gl_rotate.html
        inline fn fromRotation4x4(angle: Scalar, axis: Vec3) Self {
            const cos_angle = @cos(angle);
            const sin_angle = @sin(angle);
            const cos_value = 1.0 - cos_angle;

            const xx = axis.x() * axis.x();
            const xy = axis.x() * axis.y();
            const xz = axis.x() * axis.z();
            const yy = axis.y() * axis.y();
            const yz = axis.y() * axis.z();
            const zz = axis.z() * axis.z();
            const sx = sin_angle * axis.x();
            const sy = sin_angle * axis.y();
            const sz = sin_angle * axis.z();

            return .{ .e = .{
                RowVec.init(xx * cos_value + cos_angle, xy * cos_value + sz, xz * cos_value - sy, 0),
                RowVec.init(xy * cos_value - sz, yy * cos_value + cos_angle, yz * cos_value + sx, 0),
                RowVec.init(xz * cos_value + sy, yz * cos_value - sx, zz * cos_value + cos_angle, 0),
                RowVec.init(0, 0, 0, 1),
            } };
        }

        /// Compute 4x4 column-major homogeneous 3D rotation matrix from given euler_angle
        /// zyx order, mat = (z * y * x)
        pub inline fn fromEulerAngles(euler_angle: Vec3) M4 {
            const cos_euler = euler_angle.cos();
            const sin_euler = euler_angle.sin();

            return .{ .e = .{
                RowVec.init(
                    cos_euler.z() * cos_euler.y(),
                    sin_euler.z() * cos_euler.y(),
                    -sin_euler.y(),
                    0,
                ),
                RowVec.init(
                    cos_euler.z() * sin_euler.y() * sin_euler.x() - sin_euler.z() * cos_euler.x(),
                    sin_euler.z() * sin_euler.y() * sin_euler.x() + cos_euler.z() * cos_euler.x(),
                    cos_euler.y() * sin_euler.x(),
                    0,
                ),
                RowVec.init(
                    cos_euler.z() * sin_euler.y() * cos_euler.x() + sin_euler.z() * sin_euler.x(),
                    sin_euler.z() * sin_euler.y() * cos_euler.x() - cos_euler.z() * sin_euler.x(),
                    cos_euler.y() * cos_euler.x(),
                    0,
                ),
                RowVec.init(0, 0, 0, 1),
            } };
        }

        /// Compute 4x4 column-major homogeneous 3D scale matrix
        inline fn fromScale4x4(vec: Vec3) Self {
            return .init(
                RowVec.init(vec.x(), 0, 0, 0),
                RowVec.init(0, vec.y(), 0, 0),
                RowVec.init(0, 0, vec.z(), 0),
                RowVec.init(0, 0, 0, 1),
            );
        }

        /// Applies rotation and returns a new matrix
        pub inline fn rotate4x4(self: Self, angle: Scalar, axis: TransVec) Self {
            return Self.mul(fromRotation(angle, axis), self);
        }

        // Applies 3D translation, scale and rotation and returns a new matrix
        inline fn transformation4x4(translationv: Vec3, rotationv: Vec3, scalev: Vec3) Self {
            const cos_rot = rotationv.cos();
            const sin_rot = rotationv.sin();

            return .{
                .e = .{
                    RowVec.init(
                        cos_rot.z() * cos_rot.y() * scalev.x(),
                        sin_rot.z() * cos_rot.y() * scalev.x(),
                        -sin_rot.y() * scalev.x(),
                        0,
                    ),
                    RowVec.init(
                        (cos_rot.z() * sin_rot.y() * sin_rot.x() - sin_rot.z() * cos_rot.x()) * scalev.y(),
                        (sin_rot.z() * sin_rot.y() * sin_rot.x() + cos_rot.z() * cos_rot.x()) * scalev.y(),
                        cos_rot.y() * sin_rot.x() * scalev.y(),
                        0,
                    ),
                    RowVec.init(
                        (cos_rot.z() * sin_rot.y() * cos_rot.x() + sin_rot.z() * sin_rot.x()) * scalev.z(),
                        (sin_rot.z() * sin_rot.y() * cos_rot.x() - cos_rot.z() * sin_rot.x()) * scalev.z(),
                        cos_rot.y() * cos_rot.x() * scalev.z(),
                        0,
                    ),
                    RowVec.v3To4(translationv, 1.0),
                },
            };
        }

        /// Extract vec3 translation from matrix
        inline fn getTranslation4x4(self: Self) Vec3 {
            return self.e[3].swizzle("xyz");
        }

        /// Extract rotation euler angle from matrix
        inline fn getRotation4x4(self: Self) Vec3 {
            const col_a = self.e[0].swizzle("xyz").norm();
            const col_b = self.e[1].swizzle("xyz").norm();
            const col_c = self.e[2].swizzle("xyz").norm();

            const theta_x = std.math.atan2(col_b.z(), col_c.z());
            const c2 = @sqrt(col_a.x() * col_a.x() + col_a.y() * col_a.y());
            const theta_y = std.math.atan2(-col_a.z(), c2);
            const s1 = @sin(theta_x);
            const c1 = @cos(theta_x);
            const theta_z = std.math.atan2(s1 * col_c.x() - c1 * col_b.x(), c1 * col_b.y() - s1 * col_c.y());

            return Vec3.init(theta_x, theta_y, theta_z);
        }

        inline fn getRotationUniformScale4x4(self: Self) Vec3 {
            const len = self.e[0].swizzle("xyz").len();
            const col_a = self.e[0].swizzle("xyz").divScalar(len);
            const col_b = self.e[1].swizzle("xyz").divScalar(len);
            const col_c = self.e[2].swizzle("xyz").divScalar(len);

            const theta_x = std.math.atan2(col_b.z(), col_c.z());
            const c2 = @sqrt(col_a.x() * col_a.x() + col_a.y() * col_a.y());
            const theta_y = std.math.atan2(-col_a.z(), c2);
            const s1 = @sin(theta_x);
            const c1 = @cos(theta_x);
            const theta_z = std.math.atan2(s1 * col_c.x() - c1 * col_b.x(), c1 * col_b.y() - s1 * col_c.y());

            return Vec3.init(theta_x, theta_y, theta_z);
        }

        pub inline fn toEulerAngles(self: M4) Vec3 {
            return self.getRotation();
        }

        /// Extract vec3 scale from matrix
        inline fn getScale4x4(self: Self) Vec3 {
            return Vec3.init(
                self.e[0].swizzle("xyz").len(),
                self.e[1].swizzle("xyz").len(),
                self.e[2].swizzle("xyz").len(),
            );
        }
        //////////////////////////////////////////////////////////////////////////////////
    };
}

test "determinant" {
    // Mat2x2
    {
        const Mat2x2 = GenericMatrix(2, f32);
        const vec2 = GenericVector(2, f32).init;

        const a = Mat2x2.init(
            vec2(3, 4),
            vec2(8, 6),
        );

        try testing.expectEqual(-14, a.determinant());
    }
    // Mat3x3
    {
        const Mat3x3 = GenericMatrix(3, f32);
        const vec3 = GenericVector(3, f32).init;

        const a = Mat3x3.init(
            vec3(6, 4, 2),
            vec3(1, -2, 8),
            vec3(1, 5, 7),
        );

        try testing.expectEqual(-306, a.determinant());
    }
    // Mat4x4
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec4 = GenericVector(4, f32).init;

        const a = Mat4x4.init(
            Vec4(8, 7, 3, 9),
            Vec4(73, 9, 4, 1),
            Vec4(2, 1, 7, 4),
            Vec4(34, 2, 8, 7),
        );

        try testing.expectEqual(-19316, a.determinant());
    }
}

test "transpose" {
    // Mat3x3
    {
        const Mat3x3 = GenericMatrix(3, f32);
        const vec3 = GenericVector(3, f32).init;

        const a = Mat3x3.init(
            vec3(10, -5, 6),
            vec3(0, -1, 0),
            vec3(-1, 6, -4),
        );

        const expected = Mat3x3.init(
            vec3(10, 0, -1),
            vec3(-5, -1, 6),
            vec3(6, 0, -4),
        );

        try testing.expectEqual(expected, a.transpose());
    }

    // Mat4x4
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const vec4 = GenericVector(4, f32).init;

        const a = Mat4x4.init(
            vec4(10, -5, 6, -2),
            vec4(0, -1, 0, 9),
            vec4(-1, 6, -4, 8),
            vec4(9, -8, -6, -10),
        );

        const expected = Mat4x4.init(
            vec4(10, 0, -1, 9),
            vec4(-5, -1, 6, -8),
            vec4(6, 0, -4, -6),
            vec4(-2, 9, 8, -10),
        );

        try testing.expectEqual(expected, a.transpose());
    }
}

test "row" {
    const Mat4x4 = GenericMatrix(4, f32);
    const vec4 = GenericVector(4, f32).init;

    const a = Mat4x4.init(
        vec4(10, -5, 6, -2),
        vec4(0, -1, 0, 9),
        vec4(-1, 6, -4, 8),
        vec4(9, -8, -6, -10),
    );

    try testing.expectEqual(vec4(10, 0, -1, 9), a.row(0));
}

test "col" {
    const Mat4x4 = GenericMatrix(4, f32);
    const vec4 = GenericVector(4, f32).init;

    const a = Mat4x4.init(
        vec4(10, -5, 6, -2),
        vec4(0, -1, 0, 9),
        vec4(-1, 6, -4, 8),
        vec4(9, -8, -6, -10),
    );

    try testing.expectEqual(vec4(10, -5, 6, -2), a.col(0));
}

test "setRow" {
    const Mat4x4 = GenericMatrix(4, f32);
    const vec4 = GenericVector(4, f32).init;

    var a = Mat4x4.identity();
    a.setRow(0, vec4(10, 0, -1, 9));

    try testing.expectEqual(vec4(10, 0, -1, 9), a.row(0));
}

test "setCol" {
    const Mat4x4 = GenericMatrix(4, f32);
    const vec4 = GenericVector(4, f32).init;

    var a = Mat4x4.identity();
    a.setCol(0, vec4(10, -5, 6, -2));

    try testing.expectEqual(vec4(10, -5, 6, -2), a.col(0));
}

test "diagonal" {
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec4 = GenericVector(4, f32);

        const a = Mat4x4.diagonal(Mat4x4.ColVec.init(1, 2, 3, 4));
        try std.testing.expectEqual(Mat4x4.init(
            Vec4.init(1, 0, 0, 0),
            Vec4.init(0, 2, 0, 0),
            Vec4.init(0, 0, 3, 0),
            Vec4.init(0, 0, 0, 4),
        ), a);
    }
    {
        const Mat3x3 = GenericMatrix(3, f32);
        const Vec3 = GenericVector(3, f32);

        const a = Mat3x3.diagonal(Mat3x3.ColVec.init(1, 2, 3));
        try std.testing.expectEqual(Mat3x3.init(
            Vec3.init(1, 0, 0),
            Vec3.init(0, 2, 0),
            Vec3.init(0, 0, 3),
        ), a);
    }
}

test "mul" {
    const Mat4x4 = GenericMatrix(4, f32);
    const Vec4 = GenericVector(4, f32);

    // Mat4x4
    {
        const a = Mat4x4.init(
            Vec4.init(0, 1, 2, 3),
            Vec4.init(4, 5, 6, 7),
            Vec4.init(8, 9, 10, 11),
            Vec4.init(12, 13, 14, 15),
        );
        const b = Mat4x4.init(
            Vec4.init(4, 5, 6, 7),
            Vec4.init(1, 2, 3, 4),
            Vec4.init(9, 10, 11, 12),
            Vec4.init(-1, -2, -3, -4),
        );

        const expected = Mat4x4.init(
            Vec4.init(152, 174, 196, 218),
            Vec4.init(80, 90, 100, 110),
            Vec4.init(272, 314, 356, 398),
            Vec4.init(-80, -90, -100, -110),
        );
        try testing.expectEqual(expected, Mat4x4.mul(a, b));
    }

    // Mat4x4 x Vec4
    {
        const a = Mat4x4.init(
            Vec4.init(1, 18, 102, 79),
            Vec4.init(-1, -30, 46, -47),
            Vec4.init(-120, 76, -56, 74),
            Vec4.init(102, -9, 26, -37),
        );
        const b = Vec4.init(7, -7, -3, -8);

        try testing.expectEqual(Vec4.init(-442, 180, 352, 956), Mat4x4.mul(a, b));
    }
}

test "add" {
    const Mat4x4 = GenericMatrix(4, f32);
    const Vec4 = GenericVector(4, f32);

    const a = Mat4x4.init(
        Vec4.init(0, 1, 2, 3),
        Vec4.init(4, 5, 6, 7),
        Vec4.init(8, 9, 10, 11),
        Vec4.init(12, 13, 14, 15),
    );
    const b = Mat4x4.init(
        Vec4.init(0, 1, 2, 3),
        Vec4.init(4, 5, 6, 7),
        Vec4.init(8, 9, 10, 11),
        Vec4.init(12, 13, 14, 15),
    );

    try testing.expectEqual(a.add(b), Mat4x4.init(
        Vec4.init(0, 2, 4, 6),
        Vec4.init(8, 10, 12, 14),
        Vec4.init(16, 18, 20, 22),
        Vec4.init(24, 26, 28, 30),
    ));
}

test "sub" {
    const Mat4x4 = GenericMatrix(4, f32);
    const Vec4 = GenericVector(4, f32);

    const a = Mat4x4.init(
        Vec4.init(0, 1, 2, 3),
        Vec4.init(4, 5, 6, 7),
        Vec4.init(8, 9, 10, 11),
        Vec4.init(12, 13, 14, 15),
    );
    const b = Mat4x4.init(
        Vec4.init(0, 1, 2, 3),
        Vec4.init(4, 5, 6, 7),
        Vec4.init(8, 9, 10, 11),
        Vec4.init(12, 13, 14, 15),
    );

    try testing.expectEqual(a.sub(b), Mat4x4.init(
        Vec4.init(0, 0, 0, 0),
        Vec4.init(0, 0, 0, 0),
        Vec4.init(0, 0, 0, 0),
        Vec4.init(0, 0, 0, 0),
    ));
}

test "mulScalar" {
    const Mat4x4 = GenericMatrix(4, f32);
    const Vec4 = GenericVector(4, f32);

    const a = Mat4x4.init(
        Vec4.init(0, 1, 2, 3),
        Vec4.init(4, 5, 6, 7),
        Vec4.init(8, 9, 10, 11),
        Vec4.init(12, 13, 14, 15),
    );

    try testing.expectEqual(a.mulScalar(2), Mat4x4.init(
        Vec4.init(0, 2, 4, 6),
        Vec4.init(8, 10, 12, 14),
        Vec4.init(16, 18, 20, 22),
        Vec4.init(24, 26, 28, 30),
    ));
}

test "addScalar" {
    const Mat4x4 = GenericMatrix(4, f32);
    const Vec4 = GenericVector(4, f32);

    const a = Mat4x4.init(
        Vec4.init(0, 1, 2, 3),
        Vec4.init(4, 5, 6, 7),
        Vec4.init(8, 9, 10, 11),
        Vec4.init(12, 13, 14, 15),
    );

    try testing.expectEqual(a.addScalar(2), Mat4x4.init(
        Vec4.init(2, 3, 4, 5),
        Vec4.init(6, 7, 8, 9),
        Vec4.init(10, 11, 12, 13),
        Vec4.init(14, 15, 16, 17),
    ));
}

test "subScalar" {
    const Mat4x4 = GenericMatrix(4, f32);
    const Vec4 = GenericVector(4, f32);

    const a = Mat4x4.init(
        Vec4.init(0, 1, 2, 3),
        Vec4.init(4, 5, 6, 7),
        Vec4.init(8, 9, 10, 11),
        Vec4.init(12, 13, 14, 15),
    );

    try testing.expectEqual(a.subScalar(2), Mat4x4.init(
        Vec4.init(-2, -1, 0, 1),
        Vec4.init(2, 3, 4, 5),
        Vec4.init(6, 7, 8, 9),
        Vec4.init(10, 11, 12, 13),
    ));
}

test "extractRotation" {
    // Mat4x4 - uniform scale
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec3 = GenericVector(3, f32);

        const a = Mat4x4.fromEulerAngles(Vec3.init(0.785398, -0.0872665, 0.349066));
        try std.testing.expectEqual(Vec3.init(0.78539795, -0.0872665, 0.34906596), a.getRotation());
    }
}

test "rotate" {
    // Mat4x4
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec3 = GenericVector(3, f32);

        const a = Mat4x4.identity().rotate(0.785398, Vec3.init(0, 1, 0));
        try std.testing.expectEqual(Vec3.init(0, 0.785398, 0), a.getRotation());
    }

    // Mat3x3
    {
        const Mat3x3 = GenericMatrix(3, f32);

        const a = Mat3x3.identity().rotate(0.785398);
        try std.testing.expectEqual(7.8539836e-1, a.getRotation());
    }
}

test "transformation" {
    // Mat4x4
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec3 = GenericVector(3, f32);

        // uniform scale
        {
            const position = Vec3.init(20, 40, -50);
            const rotation = Vec3.init(0.78539795, -0.0872665, 0.34906596);
            const scale = Vec3.init(4, 4, 4);

            const a = Mat4x4.transformation(position, rotation, scale);
            try std.testing.expectEqual(position, a.getTranslation());
            try std.testing.expectEqual(rotation, a.getRotationUniformScale());
            try std.testing.expectEqual(scale, a.getScale());
        }

        // non-uniform scale
        {
            const position = Vec3.init(20, 40, -50);
            const rotation = Vec3.init(0.78539795, -0.0872665, 0.34906596);
            const scale = Vec3.init(4, 4, 2);

            const a = Mat4x4.transformation(position, rotation, scale);
            try std.testing.expectEqual(position, a.getTranslation());
            try std.testing.expectEqual(rotation, a.getRotation());
            try std.testing.expectEqual(scale, a.getScale());
        }
    }
}

test "inverse" {
    {
        const Mat2x2 = GenericMatrix(2, f32);
        const Vec2 = Mat2x2.RowVec;

        const a = Mat2x2.init(Vec2.init(4, 2), Vec2.init(7, 6)).inverse();
        try std.testing.expectEqual(Mat2x2.init(Vec2.init(0.6, -0.2), Vec2.init(-0.7, 0.4)), a);
    }

    {
        const Mat3x3 = GenericMatrix(3, f32);
        const Vec3 = Mat3x3.RowVec;

        const a = Mat3x3.init(Vec3.init(2, 5, 1), Vec3.init(5, 8, 9), Vec3.init(4, 7, 3)).inverse();
        try std.testing.expectEqual(Mat3x3.init(
            Vec3.init(-1.3000001e0, -2.6666668e-1, 1.2333333e0),
            Vec3.init(7.0000005e-1, 6.666667e-2, -4.3333337e-1),
            Vec3.init(1.0000001e-1, 2.0000002e-1, -3e-1),
        ), a);
    }

    // 4x4
    // inverse transformation no scale
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec3 = GenericVector(3, f32);
        const Vec4 = GenericVector(4, f32);

        const a = Mat4x4.transformation(Vec3.init(-8.9, -10.2, -11.4), Vec3.init(0.5, 1.5, 1.0), Vec3.init(1, 1, 1));
        const a_inv = a.invTransUnitScale();

        const a_vec = Vec4.init(2, 3, 4, 1);
        const vec = a.mul(a_vec);

        try std.testing.expect(a_inv.mul(vec).eqlApprox(a_vec, 0.00001));
    }
    // inverse transformation non-unform scale
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec3 = GenericVector(3, f32);
        const Vec4 = GenericVector(4, f32);

        const a = Mat4x4.transformation(Vec3.init(-8.9, -10.2, -11.4), Vec3.init(0.5, 1.5, 1.0), Vec3.init(90, 180, 120));
        const a_inv = a.invTrans();

        const a_vec = Vec4.init(2, 3, 4, 1);
        const vec = a.mul(a_vec);

        try std.testing.expect(a_inv.mul(vec).eqlApprox(a_vec, 0.000001));
    }
    // inverse
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec3 = GenericVector(3, f32);
        const Vec4 = GenericVector(4, f32);

        const a = Mat4x4.transformation(Vec3.init(-8.9, -10.2, -11.4), Vec3.init(0.5, 1.5, 1.0), Vec3.init(10, 8, 12));
        const a_inv = a.inverse();

        const a_vec = Vec4.init(2, 3, 4, 1);
        const vec = a.mul(a_vec);

        try std.testing.expect(a_inv.mul(vec).eqlApprox(a_vec, 0.00001));
    }
    {
        const Mat4x4 = GenericMatrix(4, f32);
        const Vec4 = Mat4x4.RowVec;

        const a = Mat4x4.init(
            Vec4.init(2, 5, 4, 1),
            Vec4.init(5, 8, 9, 7),
            Vec4.init(4, 7, 3, 2),
            Vec4.init(1, 6, 4, 3),
        ).inverse();

        try std.testing.expectEqual(Mat4x4.init(
            Vec4.init(-1.4925373e-2, 1.0447761e-1, 2.2388059e-1, -3.880597e-1),
            Vec4.init(-2.9850746e-2, -1.2437811e-1, 1.1442786e-1, 2.2388059e-1),
            Vec4.init(4.1791043e-1, 7.462686e-2, -2.686567e-1, -1.3432835e-1),
            Vec4.init(-4.925373e-1, 1.1442786e-1, 5.4726366e-2, 1.9402985e-1),
        ), a);
    }
}

test "outer" {
    const Vec3 = GenericVector(3, f32);
    const Mat3x3 = GenericMatrix(3, f32);
    const a = Vec3.init(1.0, 2.0, 3.0);
    const b = Vec3.init(4.0, 5.0, 7.0);

    try std.testing.expectEqual(Mat3x3.outer(a, b), Mat3x3.init(
        Vec3.init(4e0, 8e0, 1.2e1),
        Vec3.init(5e0, 1e1, 1.5e1),
        Vec3.init(7e0, 1.4e1, 2.1e1),
    ));
}

test "perspectiveX" {
    const Mat4x4 = GenericMatrix(4, f32);
    {
        const proj = Mat4x4.perspectiveX(1.0, 1.0, 1.0, 10002.0);
        try std.testing.expectEqual(proj.inverse(), Mat4x4.invPerspectiveX(1.0, 1.0, 1.0, 10002.0));
    }
    {
        const proj = Mat4x4.perspectiveY(1.0, 1.0, 1.0, 10002.0);
        try std.testing.expectEqual(proj.inverse(), Mat4x4.invPerspectiveY(1.0, 1.0, 1.0, 10002.0));
    }
}

test "projection2D" {
    const Mat4x4 = GenericMatrix(4, f32);
    {
        const proj = Mat4x4.projection2D(-5, 5, -5, 5, 0.1, 102.0);
        try std.testing.expectEqual(
            Mat4x4.identity(),
            proj.mul(Mat4x4.invProjection2D(-5, 5, -5, 5, 0.1, 102.0)),
        );
    }
    {
        const proj = Mat4x4.projection2D(0, 1024, 620, 0, 0.1, 102.0);
        try std.testing.expectEqual(
            Mat4x4.identity(),
            proj.mul(Mat4x4.invProjection2D(0, 1024, 620, 0, 0.1, 102.0)),
        );
    }
}
