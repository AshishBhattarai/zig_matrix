const std = @import("std");
const testing = std.testing;

pub fn GenericVector(comptime dim_i: comptime_int, comptime Scalar: type) type {
    ValidateScalarType(Scalar);

    return extern struct {
        const Self = @This();
        const Elements = @Vector(dim, Scalar);
        pub const dim = dim_i;
        pub const dim_row = dim_i;
        pub const dim_col = 1;

        elements: Elements,

        pub usingnamespace switch (dim) {
            2 => struct {
                pub inline fn init(xv: Scalar, yv: Scalar) Self {
                    return .{ .elements = [2]Scalar{ xv, yv } };
                }

                pub inline fn x(self: Self) Scalar {
                    return self.elements[0];
                }

                pub inline fn y(self: Self) Scalar {
                    return self.elements[1];
                }

                pub inline fn rotate(self: Self, ang: Scalar) Self {
                    const sin_angle = @sin(ang);
                    const cos_angle = @cos(ang);
                    return .{ .elements = .{
                        cos_angle * self.x() - sin_angle * self.y(),
                        sin_angle * self.x() + cos_angle * self.y(),
                    } };
                }

                pub inline fn orthogonal(self: Self) Self {
                    return .{ .elements = .{ -self.elements[1], self.elements[0] } };
                }

                pub inline fn cross(a: Self, b: Self) Scalar {
                    return a.elements[0] * b.elements[1] - a.elements[1] * b.elements[0];
                }
            },
            3 => struct {
                pub inline fn init(xv: Scalar, yv: Scalar, zv: Scalar) Self {
                    return .{ .elements = [3]Scalar{ xv, yv, zv } };
                }

                pub inline fn x(self: Self) Scalar {
                    return self.elements[0];
                }

                pub inline fn y(self: Self) Scalar {
                    return self.elements[1];
                }

                pub inline fn z(self: Self) Scalar {
                    return self.elements[2];
                }

                pub inline fn cross(a: Self, b: Self) Self {
                    const tmp0 = a.swizzle("yzx").elements;
                    const tmp1 = b.swizzle("zxy").elements;
                    const tmp2 = a.swizzle("zxy").elements;
                    const tmp3 = b.swizzle("yzx").elements;

                    return .{ .elements = (tmp0 * tmp1) - (tmp2 * tmp3) };
                }

                pub inline fn toEculidean(self: Self) Self {
                    const vx = self.elements[2] * @sin(self.elements[0]) * @cos(self.elements[1]);
                    const vy = self.elements[2] * @sin(self.elements[0]) * @sin(self.elements[1]);
                    const vz = self.elements[2] * @cos(self.elements[0]);

                    return .{ .elements = .{ vx, vy, vz } };
                }

                pub inline fn toEculideanDir(self: Self) Self {
                    const vx = @sin(self.elements[0]) * @cos(self.elements[1]);
                    const vy = @sin(self.elements[0]) * @sin(self.elements[1]);
                    const vz = @cos(self.elements[0]);

                    return .{ .elements = .{ vx, vy, vz } };
                }

                // x - latitude, y - longitude, r - radius
                pub inline fn toPolar(self: Self) Self {
                    const l = self.len();
                    const azimuth = std.math.acos(self.elements[2] / l);
                    const polar = std.math.atan2(self.elements[1], self.elements[0]);

                    return .{ .elements = .{ azimuth, polar, l } };
                }

                const sqrt_inv_3 = @sqrt(1.0 / 3.0);
                // expects normalized input
                pub inline fn orthogonal(self: Self) Self {
                    const cond = @abs(self.swizzle("xxx").elements) >= Self.splat(sqrt_inv_3).elements;
                    const tmp0 = Self.select(Self.init(self.y(), -self.x(), 0.0), Self.init(0.0, self.z(), -self.y()), cond);
                    return tmp0.norm();
                }
            },
            4 => struct {
                pub inline fn init(xv: Scalar, yv: Scalar, zv: Scalar, sw: Scalar) Self {
                    return .{ .elements = [4]Scalar{ xv, yv, zv, sw } };
                }

                pub inline fn fromVec3(vec3: GenericVector(3, Scalar), new_elem: Scalar) Self {
                    return Self.init(vec3.x(), vec3.y(), vec3.z(), new_elem);
                }

                pub inline fn x(self: Self) Scalar {
                    return self.elements[0];
                }

                pub inline fn y(self: Self) Scalar {
                    return self.elements[1];
                }

                pub inline fn z(self: Self) Scalar {
                    return self.elements[2];
                }

                pub inline fn w(self: Self) Scalar {
                    return self.elements[3];
                }
            },
            else => @compileError("Vector dimensions must be in range [2,4]"),
        };

        // arithmetic operations

        pub inline fn add(a: Self, b: Self) Self {
            return .{ .elements = a.elements + b.elements };
        }

        pub inline fn sub(a: Self, b: Self) Self {
            return .{ .elements = a.elements - b.elements };
        }

        pub inline fn mul(a: Self, b: Self) Self {
            return .{ .elements = a.elements * b.elements };
        }

        pub inline fn div(a: Self, b: Self) Self {
            return .{ .elements = a.elements / b.elements };
        }

        pub inline fn addScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements + Self.splat(s).elements };
        }

        pub inline fn subScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements - Self.splat(s).elements };
        }

        pub inline fn mulScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements * Self.splat(s).elements };
        }

        pub inline fn divScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements / Self.splat(s).elements };
        }

        // vector - scalar operations

        pub inline fn dot(a: Self, b: Self) Scalar {
            return @reduce(.Add, a.elements * b.elements);
        }

        // normalized dot product
        pub inline fn normDot(a: Self, b: Self) Scalar {
            const adota = a.dot(a);
            const bdotb = a.dot(b);
            const adotb = a.dot(b);
            const is_zero = adota == 0 or bdotb == 0;
            const ret = adotb / @sqrt(a.dot(a) * b.dot(b));
            return if (is_zero) 0 else ret;
        }

        pub inline fn normDotUc(a: Self, b: Self) Scalar {
            return a.dot(b) / @sqrt(a.dot(a) * b.dot(b));
        }

        pub inline fn sqrLen(a: Self) Scalar {
            return @reduce(.Add, a.elements * a.elements);
        }

        pub inline fn len(self: Self) Scalar {
            return @sqrt(self.sqrLen());
        }

        pub inline fn eql(a: Self, b: Self) bool {
            return @reduce(.And, a.elements == b.elements);
        }

        pub inline fn neql(a: Self, b: Self) bool {
            return @reduce(.And, a.elements != b.elements);
        }

        pub inline fn eqlApprox(a: Self, b: Self, tolerance: Scalar) bool {
            return @reduce(.And, @abs(a.elements - b.elements) <= @as(@Vector(dim, Scalar), @splat(tolerance)));
        }

        // a < b ?
        pub inline fn lt(a: Self, b: Self) bool {
            return @reduce(.And, a.elements < b.elements);
        }

        // a <= b ?
        pub inline fn lte(a: Self, b: Self) bool {
            return @reduce(.And, a.elements <= b.elements);
        }

        // a > b ?
        pub inline fn gt(a: Self, b: Self) bool {
            return @reduce(.And, a.elements > b.elements);
        }

        // a >= b ?
        pub inline fn gte(a: Self, b: Self) bool {
            return @reduce(.And, a.elements >= b.elements);
        }

        // vector operations

        pub inline fn norm(self: Self) Self {
            const length = self.len();
            const normalized = self.divScalar(length);
            const pred: @Vector(dim, bool) = @splat(length < std.math.floatEps(Scalar));
            const zero: @Vector(dim, Scalar) = @splat(0);
            return .{ .elements = @select(Scalar, pred, zero, normalized.elements) };
        }

        // unchecked norm
        pub inline fn normUc(self: Self) Self {
            const length = self.len();
            return self.divScalar(length);
        }

        pub inline fn sqrDist(a: Self, b: Self) Scalar {
            return Self.sqrLen(a.sub(b));
        }

        pub inline fn dist(a: Self, b: Self) Scalar {
            return Self.len(a.sub(b));
        }

        pub inline fn floor(a: Self) Self {
            return .{ .elements = @floor(a.elements) };
        }

        pub inline fn ceil(a: Self) Self {
            return .{ .elements = @ceil(a.elements) };
        }

        pub inline fn min(a: Self, b: Self) Self {
            return .{ .elements = @min(a.elements, b.elements) };
        }

        pub inline fn clamp(a: Self, lower: Self, upper: Self) Self {
            return .{ .elements = @min(@max(lower.elements, a.elements), upper.elements) };
        }

        pub inline fn max(a: Self, b: Self) Self {
            return .{ .elements = @max(a.elements, b.elements) };
        }

        pub inline fn minElem(a: Self) Scalar {
            return a.min(a.swizzle("yzx")).min(a.swizzle("zxy")).elements[0];
        }

        pub inline fn minElemIdx(a: Self) u8 {
            const min_elems = a.min(a.swizzle("yzx")).min(a.swizzle("zxy")).elements;
            const mask: u3 = @bitCast(a.elements == min_elems);
            return @ctz(mask);
        }

        pub inline fn maxElem(a: Self) Scalar {
            return a.max(a.swizzle("yzx")).max(a.swizzle("zxy")).elements[0];
        }

        pub inline fn maxElemIdx(a: Self) u8 {
            const max_elems = a.max(a.swizzle("yzx")).max(a.swizzle("zxy")).elements;
            const mask: u3 = @bitCast(a.elements == max_elems);
            return @ctz(mask);
        }

        // a * (1 -t) + bt
        pub inline fn lerp(a: Self, b: Self, t: Scalar) Self {
            return .{ .elements = a.elements + splat(t).elements * (b.elements - a.elements) };
        }

        pub inline fn inverse(self: Self) Self {
            return Self.splat(1.0).div(self);
        }

        pub inline fn negate(self: Self) Self {
            return self.mulScalar(-1.0);
        }

        // atan2(a.b, axb) for more stable angle near 0 ?

        // assumes unit vectors
        pub inline fn angle(a: Self, b: Self) Scalar {
            return std.math.acos(a.dot(b));
        }

        // normalize and compute angle
        pub inline fn normAngle(a: Self, b: Self) Scalar {
            return std.math.acos(a.normDot(b));
        }

        pub inline fn normAngleUc(a: Self, b: Self) Scalar {
            return std.math.acos(a.normDotUc(b));
        }

        pub inline fn cos(self: Self) Self {
            return .{ .elements = @cos(self.elements) };
        }

        pub inline fn sin(self: Self) Self {
            return .{ .elements = @sin(self.elements) };
        }

        pub inline fn abs(self: Self) Self {
            return .{ .elements = @abs(self.elements) };
        }

        pub inline fn sign(self: Self) Self {
            const zero: Elements = @splat(0.0);
            const pos: Elements = @splat(1.0);
            const neg: Elements = @splat(-1.0);

            const pos_sign: Elements = @select(Scalar, self.elements > zero, pos, zero);
            const neg_sign: Elements = @select(Scalar, self.elements < zero, neg, zero);

            return .{ .elements = pos_sign + neg_sign };
        }

        // sign that doesn't return zero
        pub inline fn signnz(self: Self) Self {
            const zero: Elements = @splat(0.0);
            const pos: Elements = @splat(1.0);
            const neg: Elements = @splat(-1.0);

            return .{ .elements = @select(Scalar, self.elements < zero, neg, pos) };
        }

        // utilities

        pub inline fn shuffle(a: Self, b: Self, mask: @Vector(dim, i32)) Self {
            return .{ .elements = @shuffle(Scalar, a.elements, b.elements, mask) };
        }

        pub inline fn shuffleN(a: Self, b: Self, comptime N: comptime_int, mask: @Vector(N, i32)) GenericVector(N, Scalar) {
            return .{ .elements = @shuffle(Scalar, a.elements, b.elements, mask) };
        }

        pub inline fn select(a: Self, b: Self, pred: @Vector(dim, bool)) Self {
            return .{ .elements = @select(Scalar, pred, a.elements, b.elements) };
        }

        pub inline fn splat(scalar: Scalar) Self {
            return .{ .elements = @splat(scalar) };
        }

        pub inline fn fract(a: Self) Self {
            const ipart = @trunc(a.elements);
            return .{ .elements = a.elements - ipart };
        }

        pub inline fn fromSlice(data: []const Scalar) Self {
            return .{ .elements = data[0..dim].* };
        }

        pub inline fn isInf(self: Self) bool {
            var is_inf = false;
            inline for (0..dim) |i| {
                is_inf = std.math.isInf(self.elements[i]) or is_inf;
            }
            return is_inf;
        }

        pub inline fn isNan(self: Self) bool {
            var is_nan = false;
            inline for (0..dim) |i| {
                is_nan = std.math.isNan(self.elements[i]) or is_nan;
            }
            return is_nan;
        }

        pub inline fn toFloat(self: Self, comptime NewScalar: type) GenericVector(dim_i, NewScalar) {
            ValidateScalarType(NewScalar);
            if (@typeInfo(Scalar) == .float) {
                return .{ .elements = @floatCast(self.elements) };
            } else {
                return .{ .elements = @floatFromInt(self.elements) };
            }
        }

        pub inline fn toInt(self: Self, comptime NewScalar: type) GenericVector(dim_i, NewScalar) {
            ValidateScalarType(NewScalar);
            if (@typeInfo(Scalar) == .int) {
                return .{ .elements = @intCast(self.elements) };
            } else {
                return .{ .elements = @intFromFloat(self.elements) };
            }
        }

        pub inline fn toType(self: Self, comptime NewScalar: type) GenericVector(dim_i, NewScalar) {
            if (@typeInfo(NewScalar) == .int) {
                return self.toInt(NewScalar);
            } else {
                return self.toFloat(NewScalar);
            }
        }

        pub inline fn swizzle(self: Self, comptime components: []const u8) GenericVector(components.len, Scalar) {
            comptime {
                if (components.len < 2 or components.len > 4) {
                    @compileError("Component length must be in range [2,4]");
                }
                for (components) |component| {
                    if (dim <= charToElementIdx(component)) {
                        @compileError("Vector dimension out of bound.");
                    }
                }
            }

            @setEvalBranchQuota(100000);
            const swizzle_elems = comptime switch (components.len) {
                2 => [2]Scalar{
                    charToElementIdx(components[0]),
                    charToElementIdx(components[1]),
                },
                3 => [3]Scalar{
                    charToElementIdx(components[0]),
                    charToElementIdx(components[1]),
                    charToElementIdx(components[2]),
                },
                4 => [4]Scalar{
                    charToElementIdx(components[0]),
                    charToElementIdx(components[1]),
                    charToElementIdx(components[2]),
                    charToElementIdx(components[3]),
                },
                else => unreachable,
            };

            return .{ .elements = @shuffle(Scalar, self.elements, undefined, swizzle_elems) };
        }

        inline fn charToElementIdx(comptime char: u8) comptime_int {
            if (char == 'x') return 0;
            if (char == 'y') return 1;
            if (char == 'z') return 2;
            if (char == 'w') return 3;

            @compileError("Invalid swizzle dimension, expected 'x' or 'y' or 'z' or 'w'");
        }
    };
}
fn ValidateScalarType(comptime Scalar: type) void {
    if (@typeInfo(Scalar) != .float and @typeInfo(Scalar) != .int) {
        @compileError("Vectors cannot be of type " ++ @typeName(Scalar));
    }
}

test "init" {
    const vec4 = GenericVector(4, f32).init(0, 1, 2, 3);
    try testing.expectEqual(0, vec4.x());
    try testing.expectEqual(1, vec4.y());
    try testing.expectEqual(2, vec4.z());
    try testing.expectEqual(3, vec4.w());
}

test "swizzle" {
    const vec2 = GenericVector(2, f32).init(1, 2);
    const vec3 = vec2.swizzle("xyx");
    try testing.expectEqual(GenericVector(3, f32).init(1, 2, 1), vec3);

    const vec4 = vec2.swizzle("yyxx");
    try testing.expectEqual(GenericVector(4, f32).init(2, 2, 1, 1), vec4);
    try testing.expectEqual(vec4.swizzle("wx"), vec2);
}

test "vec3 cross" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(3, 0, 0);
    const b = Vec3.init(1, 2, 0);
    try testing.expectEqual(Vec3.init(0, 0, 6), Vec3.cross(a, b));

    const c = Vec3.init(1.61, -0.41, 1);
    const d = Vec3.init(-2.42, 1.4, 1.78);
    try testing.expectEqual(Vec3.init(-2.1297998, -5.2858, 1.2617999), Vec3.cross(c, d));
}

test "vec2 rotate" {
    const Vec2 = GenericVector(2, f32);
    const a = Vec2.init(2.5, 2.5);
    try testing.expectEqual(Vec2.init(0.99539256, 3.3925202), a.rotate(0.5));
    try testing.expectEqual(Vec2.init(1.6106732, -3.1473372), a.rotate(4.4));
}

test "splat" {
    // vec2
    {
        const Vec4 = GenericVector(2, f32);
        try testing.expectEqual(Vec4.init(2, 2), Vec4.splat(2));
    }

    // vec3
    {
        const Vec4 = GenericVector(3, f32);
        try testing.expectEqual(Vec4.init(3, 3, 3), Vec4.splat(3));
    }

    // vec4
    {
        const Vec4 = GenericVector(4, f32);
        try testing.expectEqual(Vec4.init(5, 5, 5, 5), Vec4.splat(5));
    }
}

test "fromSlice" {
    // vec2
    {
        const Vec2 = GenericVector(2, f32);
        try testing.expectEqual(Vec2.init(2.5, 0.2), Vec2.fromSlice(&.{ 2.5, 0.2 }));
    }
    // vec3
    {
        const Vec2 = GenericVector(3, f32);
        try testing.expectEqual(Vec2.init(2.5, 0.2, -20), Vec2.fromSlice(&.{ 2.5, 0.2, -20 }));
    }
    // vec4
    {
        const Vec4 = GenericVector(4, f32);
        try testing.expectEqual(Vec4.init(2.5, 0.2, -20, 1.0), Vec4.fromSlice(&.{ 2.5, 0.2, -20, 1.0 }));
    }
}

test "add" {
    // vec2
    {
        const Vec2 = GenericVector(2, f32);
        const a = Vec2.init(1.62, 0.4);
        const b = Vec2.init(2.41, 1.4);
        try testing.expectEqual(Vec2.init(4.03, 1.8), a.add(b));
    }
    // vec3
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(4.61, -0.41, 1);
        const b = Vec3.init(-2.42, 1.4, 1.78);
        try testing.expectEqual(Vec3.init(2.19, 0.99, 2.78), a.add(b));
    }
    // vec4
    {
        const Vec4 = GenericVector(4, f32);
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66);
        try testing.expectEqual(Vec4.init(7.03, 11.809999, 2.78, 4.88), a.add(b));
    }
}

test "sub" {
    // vec2
    {
        const Vec2 = GenericVector(2, f32);
        const a = Vec2.init(1.62, 0.4);
        const b = Vec2.init(2.41, 1.4);
        try testing.expectEqual(Vec2.init(-0.7900001, -1.0), a.sub(b));
    }
    // vec3
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(4.61, -0.41, 1);
        const b = Vec3.init(-2.42, 1.4, 1.78);
        try testing.expectEqual(Vec3.init(7.03, -1.81, -0.78), a.sub(b));
    }
    // vec4
    {
        const Vec4 = GenericVector(4, f32);
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66);
        try testing.expectEqual(Vec4.init(2.19, -6.99, -0.78, -2.44), a.sub(b));
    }
}

test "dot" {
    // vec2
    {
        const Vec2 = GenericVector(2, f32);
        const a = Vec2.init(1.61, -0.41);
        const b = Vec2.init(-2.42, 1.4);
        try testing.expectEqual(-4.4702, Vec2.dot(a, b));
    }
    // vec3
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(1.61, -0.41, 1);
        const b = Vec3.init(-2.42, 1.4, 1.78);
        try testing.expectEqual(-2.6902, Vec3.dot(a, b));
    }
    // vec4
    {
        const Vec4 = GenericVector(4, f32);
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66);
        try testing.expectEqual(4.0055397e1, Vec4.dot(a, b));
    }
}

test "normDot" {
    const Vec4 = GenericVector(4, f32);
    {
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66);
        try testing.expectEqual(7.001017e-1, Vec4.normDot(a, b));
    }
    {
        const a = Vec4.init(4.61, 2.41, 1, 1.22).norm();
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66).norm();
        try testing.expectEqual(7.001018e-1, Vec4.dot(a, b));
    }
    {
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(0, 0, 0, 0);
        try testing.expectEqual(0, Vec4.normDot(a, b));
    }
}

test "normDotUc" {
    const Vec4 = GenericVector(4, f32);
    {
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66);
        try testing.expectEqual(7.001017e-1, Vec4.normDotUc(a, b));
    }
    {
        const a = Vec4.init(4.61, 2.41, 1, 1.22).norm();
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66).norm();
        try testing.expectEqual(7.001018e-1, Vec4.dot(a, b));
    }
    {
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(0, 0, 0, 0);
        try testing.expectEqual(false, Vec4.normDotUc(a, b) >= 0); // results in nan
        try testing.expectEqual(false, Vec4.normDotUc(a, b) <= 0);
    }
}

test "normAngle" {
    const Vec4 = GenericVector(4, f32);
    {
        const a = Vec4.init(4.61, 2.41, 1, 1.22);
        const b = Vec4.init(2.42, 9.4, 1.78, 3.66);
        try testing.expectEqual(7.9525626e-1, Vec4.angle(a.norm(), b.norm()));
        try testing.expectEqual(7.9525644e-1, Vec4.normAngle(a, b));
    }
}

test "len" {
    // vec2
    {
        const Vec2 = GenericVector(2, f32);
        const a = Vec2.init(1.61, -0.41);
        try testing.expectEqual(1.6613849, a.len());
    }
    // vec3
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(1.61, -0.41, 2.44);
        try testing.expectEqual(2.9519148, a.len());
    }
    // vec4
    {
        const Vec4 = GenericVector(4, f32);
        const a = Vec4.init(1.61, -0.41, 2.44, -6.22);
        try testing.expectEqual(6.884925562, a.len());
    }
}

test "norm" {
    // vec3
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(5, 7, 10);
        try testing.expectEqual(Vec3.init(3.7904903e-1, 5.306686e-1, 7.5809807e-1), a.norm());
    }
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(0, 0, 0);
        try testing.expectEqual(Vec3.splat(0), a.norm());
    }
}

test "normUc" {
    // vec3
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(5, 7, 10);
        try testing.expectEqual(Vec3.init(3.7904903e-1, 5.306686e-1, 7.5809807e-1), a.normUc());
    }
    {
        const Vec3 = GenericVector(3, f32);
        const a = Vec3.init(0, 0, 0);
        try testing.expectEqual(false, a.normUc().gte(Vec3.splat(0))); // nan
        try testing.expectEqual(false, a.normUc().lte(Vec3.splat(0)));
    }
}

test "lerp" {
    // vec4
    {
        const Vec4 = GenericVector(4, f32);
        const a = Vec4.init(1, 1, 1, 1);
        const b = Vec4.init(10, 10, 10, 10);
        try testing.expectEqual(Vec4.init(5.5, 5.5, 5.5, 5.5), a.lerp(b, 0.5));
    }
}

test "isNan isInf" {
    const Vec4 = GenericVector(4, f32);

    const a = Vec4.init(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    const b = Vec4.init(std.math.nan(f32), std.math.nan(f32), std.math.nan(f32), std.math.nan(f32));

    try testing.expect(a.isInf());
    try testing.expect(b.isNan());
}

test "toEculidean" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(1.570, 0.295, 5);

    try testing.expectEqual(Vec3.init(4.7840095, 1.4536988, 0.0039813714), a.toEculidean());
}

test "toPolar" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(4.7840095, 1.4536988, 0.0039813714);

    try testing.expectEqual(Vec3.init(1.570, 0.295, 5), a.toPolar());
}

test "sign" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(-20.0, 0, 6.88);
    const sign = a.sign();

    try testing.expectEqual(Vec3.init(std.math.sign(a.x()), std.math.sign(a.y()), std.math.sign(a.z())), sign);
}

test "signnz" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(-20.0, 0, 6.88);
    const sign = a.signnz();

    try testing.expectEqual(Vec3.init(std.math.sign(a.x()), 1.0, std.math.sign(a.z())), sign);
}

test "minElem" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(20.0, 0, -6.88);

    const min = a.minElem();
    const min_idx = a.minElemIdx();

    try testing.expectEqual(-6.88, min);
    try testing.expectEqual(2, min_idx);
}

test "maxElem" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(20.0, 0, -6.88);

    const max = a.maxElem();
    const max_idx = a.maxElemIdx();

    try testing.expectEqual(20.0, max);
    try testing.expectEqual(0, max_idx);
}

test "orthogonal" {
    const Vec2 = GenericVector(2, f32);
    const Vec3 = GenericVector(3, f32);
    {
        const a = Vec2.init(9, 10);
        try testing.expectEqual(a.rotate(std.math.pi * 0.5), a.orthogonal());
    }
    {
        const a = Vec3.init(9, 10, 11).norm();
        try testing.expectEqual(Vec3.init(0e0, 7.399401e-1, -6.726728e-1), a.orthogonal());
    }
}

test "toInt" {
    const Vec3f = GenericVector(3, f32);
    const Vec3i = GenericVector(3, i32);
    const Vec3u = GenericVector(3, u32);

    // float to int
    {
        const vec = Vec3f.init(1.0, 2.0, 3.0);
        try testing.expectEqual(Vec3i.init(1, 2, 3), vec.toInt(i32));
    }
    // int to int
    {
        const vec = Vec3i.init(1, 2, 3);
        try testing.expectEqual(Vec3u.init(1, 2, 3), vec.toInt(u32));
    }
}

test "toFloat" {
    const Vec3f = GenericVector(3, f32);
    const Vec3d = GenericVector(3, f64);
    const Vec3u = GenericVector(3, u32);

    // int to float
    {
        const vec = Vec3u.init(2, 3, 4);
        try testing.expectEqual(Vec3f.init(2, 3, 4), vec.toFloat(f32));
    }

    // float to float
    {
        const vec = Vec3f.init(4, 5, 6);
        try testing.expectEqual(Vec3d.init(4, 5, 6), vec.toFloat(f64));
    }
}

test "clamp" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(-1, 0.5, 10);
    try testing.expectEqual(Vec3.init(0, 0.5, 1), a.clamp(Vec3.splat(0), Vec3.splat(1)));
}

test "fract" {
    const Vec3 = GenericVector(3, f32);
    const a = Vec3.init(1.0, 0.5, 1.000004);
    try testing.expectEqual(Vec3.init(0, 0.5, 0.000004053116), a.fract());
}
