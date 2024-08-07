const std = @import("std");
const testing = std.testing;

pub fn GenericVector(comptime dim_i: comptime_int, comptime Scalar: type) type {
    if (@typeInfo(Scalar) != .Float and @typeInfo(Scalar) != .Int) {
        @compileError("Vectors cannot be of type " ++ @typeName(Scalar));
    }

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
                    return .{ .elements = [2]Scalar{
                        cos_angle * self.x() - sin_angle * self.y(),
                        sin_angle * self.x() + cos_angle * self.y(),
                    } };
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

        pub inline fn sqrLen(a: Self) Scalar {
            return @reduce(.Add, a.elements * a.elements);
        }

        pub inline fn len(self: Self) Scalar {
            return @sqrt(self.sqrLen());
        }

        pub inline fn eql(a: Self, b: Self) bool {
            return @reduce(.And, a.elements == b.elements);
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
            return if (length != 0) self.divScalar(length) else self;
        }

        pub inline fn sqrDist(a: Self, b: Self) Self {
            return Self.sqrLen(a.sub(b));
        }

        pub inline fn dist(a: Self, b: Self) Scalar {
            return Self.len(a.sub(b));
        }

        pub inline fn min(a: Self, b: Self) Self {
            return .{ .elements = @min(a.elements, b.elements) };
        }

        pub inline fn max(a: Self, b: Self) Self {
            return .{ .elements = @max(a.elements, b.elements) };
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

        // assumes unit vectors
        pub inline fn angle(a: Self, b: Self) Scalar {
            return std.math.acos(a.dot(b));
        }

        pub inline fn nAngle(a: Self, b: Self) Scalar {
            return std.math.acos((a.norm().dot(b.norm())));
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

        // utilities

        pub inline fn shuffle(a: Self, b: Self, mask: @Vector(dim, i32)) Self {
            return .{ .elements = @shuffle(Scalar, a.elements, b.elements, mask) };
        }

        pub inline fn splat(scalar: Scalar) Self {
            return .{ .elements = @splat(scalar) };
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

        pub inline fn swizzle(self: Self, comptime components: []const u8) GenericVector(components.len, Scalar) {
            comptime {
                if (components.len < 2 or components.len > 4) {
                    @compileError("Swizzle dimensions must be in range [2,4]");
                }
                for (components) |component| {
                    if (component < 'w' or component > 'z') {
                        @compileError("invalid swizzle dimension, expected 'x' or 'y' or 'z' or 'w'.");
                    } else if (dim < charToElementIdx(component)) {
                        @compileError("vector dimension out of bound.");
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
            return if (char == 119) 3 else char - 120;
        }
    };
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
