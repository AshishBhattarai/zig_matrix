const std = @import("std");
const testing = std.testing;
const GenericVector = @import("vector.zig").GenericVector;
const GenericMatrix = @import("matrix.zig").GenericMatrix;

pub fn GenericQuat(comptime Scalar: type) type {
    const Vec3 = GenericVector(3, Scalar);
    const Mat3 = GenericMatrix(3, 3, Scalar);
    const Mat4 = GenericMatrix(4, 4, Scalar);

    return extern struct {
        const Self = @This();
        const Elements = @Vector(4, Scalar);

        elements: Elements,

        pub inline fn init(xv: Scalar, yv: Scalar, zv: Scalar, wv: Scalar) Self {
            return .{ .elements = [4]Scalar{ xv, yv, zv, wv } };
        }

        pub inline fn identity() Self {
            return .{ .elements = [4]Scalar{ 0.0, 0.0, 0.0, 1.0 } };
        }

        pub inline fn fromSlice(data: []const Scalar) Self {
            return .{ .elements = data[0..4].* };
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

        pub inline fn mul(a: Self, b: Self) Self {
            const xv = splat(a.elements[0]) * @shuffle(Scalar, b.elements, -b.elements, [4]i32{ 3, -3, 1, -1 });
            const yv = splat(a.elements[1]) * @shuffle(Scalar, b.elements, -b.elements, [4]i32{ 2, 3, -1, -2 });
            const zv = splat(a.elements[2]) * @shuffle(Scalar, b.elements, -b.elements, [4]i32{ -2, 0, 3, -3 });
            const wv = splat(a.elements[3]) * b.elements;
            return .{ .elements = xv + yv + zv + wv };
        }

        // pure * quat
        pub inline fn mulP(quat: Self, vec: Vec3) Self {
            const xv = splat(vec.elements[0]) * @shuffle(Scalar, quat.elements, -quat.elements, [4]i32{ 3, -3, 1, -1 });
            const yv = splat(vec.elements[1]) * @shuffle(Scalar, quat.elements, -quat.elements, [4]i32{ 2, 3, -1, -2 });
            const zv = splat(vec.elements[2]) * @shuffle(Scalar, quat.elements, -quat.elements, [4]i32{ -2, 0, 3, -3 });
            return .{ .elements = xv + yv + zv };
        }

        // assumes the quaternion is unit
        // https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
        pub inline fn rotate(self: Self, vec: Vec3) Vec3 {
            const qvec = self.getVec();
            const t = qvec.cross(vec).mulScalar(2.0);
            const qw: Vec3 = .{ .elements = @shuffle(Scalar, self.elements, undefined, [3]i32{ 3, 3, 3 }) };
            return vec.add(t.mul(qw)).add(qvec.cross(t));
        }

        // quat * quat(sin(angle*0.5), 0, 0, cos(angle*0.5))
        pub inline fn rotateX(self: Self, anglev: Scalar) Self {
            const cos_ha = @cos(anglev * 0.5);
            const sin_ha = @sin(anglev * 0.5);

            const t0 = splat(cos_ha) * self.elements;
            const t1 = splat(sin_ha) * @shuffle(Scalar, self.elements, -self.elements, [4]i32{ 3, 2, -2, -1 });

            return .{ .elements = t0 + t1 };
        }

        // quat * quat(0, sin(angle*0.5), 0, cos(angle*0.5))
        pub inline fn rotateY(self: Self, anglev: Scalar) Self {
            const cos_ha = @cos(anglev * 0.5);
            const sin_ha = @sin(anglev * 0.5);

            const t0 = splat(cos_ha) * self.elements;
            const t1 = splat(sin_ha) * @shuffle(Scalar, self.elements, -self.elements, [4]i32{ -3, 3, 0, -2 });

            return .{ .elements = t0 + t1 };
        }

        // quat * quat(0, 0, sin(angle*0.5), cos(angle*0.5))
        pub inline fn rotateZ(self: Self, anglev: Scalar) Self {
            const cos_ha = @cos(anglev * 0.5);
            const sin_ha = @sin(anglev * 0.5);

            const t0 = splat(cos_ha) * self.elements;
            const t1 = splat(sin_ha) * @shuffle(Scalar, self.elements, -self.elements, [4]i32{ -2, 0, 3, -3 });

            return .{ .elements = t0 + t1 };
        }

        // zyx order, quat = (z * y * x)
        pub inline fn fromEulerAngles(euler_angle: Vec3) Self {
            const cos_euler = euler_angle.mulScalar(0.5).cos();
            const sin_euler = euler_angle.mulScalar(0.5).sin();

            return .{ .elements = [4]Scalar{
                sin_euler.x() * cos_euler.y() * cos_euler.z() - cos_euler.x() * sin_euler.y() * sin_euler.z(),
                cos_euler.x() * sin_euler.y() * cos_euler.z() + sin_euler.x() * cos_euler.y() * sin_euler.z(),
                cos_euler.x() * cos_euler.y() * sin_euler.z() - sin_euler.x() * sin_euler.y() * cos_euler.z(),
                cos_euler.x() * cos_euler.y() * cos_euler.z() + sin_euler.x() * sin_euler.y() * sin_euler.z(),
            } };
        }

        pub inline fn toEulerAngles(self: Self) Vec3 {
            const ax = std.math.atan2(
                2 * (self.w() * self.x() + self.y() * self.z()),
                1 - 2 * (self.x() * self.x() + self.y() * self.y()),
            );
            const ay = std.math.asin(@max(-1, @min(1, 2 * (self.w() * self.y() - self.x() * self.z()))));
            const az = std.math.atan2(
                2 * (self.w() * self.z() + self.x() * self.y()),
                1 - 2 * (self.y() * self.y() + self.z() * self.z()),
            );
            return .{ .elements = [3]Scalar{ ax, ay, az } };
        }

        pub inline fn fromAxis(anglev: Scalar, axis: Vec3) Self {
            return Self.fromVec(axis.mulScalar(@sin(anglev * 0.5)), @cos(anglev * 0.5));
        }

        // assumes the quaternion to be unit
        pub inline fn toMat3(self: Self) Mat3 {
            const xx = self.x() * self.x();
            const yy = self.y() * self.y();
            const zz = self.z() * self.z();
            const xy = self.x() * self.y();
            const xz = self.x() * self.z();
            const yz = self.y() * self.z();
            const wx = self.w() * self.x();
            const wy = self.w() * self.y();
            const wz = self.w() * self.z();

            return Mat3.init(
                Mat3.RowVec.init(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)),
                Mat3.RowVec.init(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)),
                Mat3.RowVec.init(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)),
            );
        }

        pub inline fn toMat4(self: Self) Mat4 {
            return Mat4.fromMat3(self.toMat3(), Mat4.RowVec.init(0, 0, 0, 1));
        }

        pub inline fn fromMat3(mat: Mat3) Self {
            const m00 = mat.element(0, 0);
            const m11 = mat.element(1, 1);
            const m22 = mat.element(2, 2);

            const t0 = 1 + m00 - m11 - m22;
            const q0 = Elements{
                t0,
                mat.element(0, 1) + mat.element(1, 0),
                mat.element(2, 0) + mat.element(0, 2),
                mat.element(1, 2) - mat.element(2, 1),
            };
            const t1 = 1 - m00 + m11 - m22;
            const q1 = Elements{
                mat.element(0, 1) + mat.element(1, 0),
                t1,
                mat.element(1, 2) + mat.element(2, 1),
                mat.element(2, 0) - mat.element(0, 2),
            };
            const t2 = 1 - m00 - m11 + m22;
            const q2 = Elements{
                mat.element(2, 0) + mat.element(0, 2),
                mat.element(1, 2) + mat.element(2, 1),
                t2,
                mat.element(0, 1) - mat.element(1, 0),
            };
            const t3 = 1 + m00 + m11 + m22;
            const q3 = Elements{
                mat.element(1, 2) - mat.element(2, 1),
                mat.element(2, 0) - mat.element(0, 2),
                mat.element(0, 1) - mat.element(1, 0),
                t3,
            };

            const q01, const t01 = if (m00 > m11) .{ q0, t0 } else .{ q1, t1 };
            const q23, const t23 = if (m00 < -m11) .{ q2, t2 } else .{ q3, t3 };
            const q, const t = if (m22 < 0) .{ q01, t01 } else .{ q23, t23 };

            return .{ .elements = (q * splat(0.5 / @sqrt(t))) };
        }

        pub inline fn fromMat4(mat: Mat4) Self {
            return fromMat3(mat.toMat3());
        }

        pub inline fn fromVec(vec: Vec3, wv: Scalar) Self {
            return .{ .elements = [4]Scalar{ vec.elements[0], vec.elements[1], vec.elements[2], wv } };
        }

        pub inline fn getVec(self: Self) Vec3 {
            return .{ .elements = [3]Scalar{ self.elements[0], self.elements[1], self.elements[2] } };
        }

        pub inline fn lerp(a: Self, b: Self, t: Scalar) Self {
            return .{ .element = a.elements + splat(t) * (b.elements - a.elements) };
        }

        // assumes a,b are unit quaternion
        pub inline fn slerp(a: Self, b: Self, t: Scalar) Self {
            var q2 = b;
            var cos_theta = a.dot(b);
            const sign = std.math.sign(cos_theta);

            // pick the shortest path
            cos_theta = cos_theta * sign;
            q2 = q2.mulScalar(sign);

            const theta = std.math.acos(cos_theta);

            // use lerp for when the angle is close to 0, prevents zero division with sin(0)
            const w0, const w1 = if (theta < std.math.floatEps(Scalar))
                .{ 1 - t, t }
            else
                .{ @sin((1 - t) * theta) / @sin(theta), @sin(theta * t) / @sin(theta) };

            return .{ .elements = splat(w0) * a.elements + splat(w1) * b.elements };
        }

        pub inline fn dot(a: Self, b: Self) Scalar {
            return @reduce(.Add, a.elements * b.elements);
        }

        pub inline fn add(a: Self, b: Self) Self {
            return .{ .elements = a.elements + b.elements };
        }

        pub inline fn sub(a: Self, b: Self) Self {
            return .{ .elements = a.elements - b.elements };
        }

        pub inline fn eql(a: Self, b: Self) bool {
            return @reduce(.And, a.elements == b.elements);
        }

        pub inline fn eqlApprox(a: Self, b: Self, tolerance: Scalar) bool {
            var ret = true;
            inline for (0..4) |i| {
                ret = ret and (@abs(a.elements[i] - b.elements[i]) <= tolerance);
            }
            return ret;
        }

        pub inline fn inverse(self: Self) Self {
            return .{ .elements = (conjugate(self).elements / splat(sqrLen(self))) };
        }

        pub inline fn conjugate(self: Self) Self {
            return .{ .elements = self.elements * [4]Scalar{ -1, -1, -1, 1 } };
        }

        pub inline fn sqrLen(self: Self) Scalar {
            return @reduce(.Add, self.elements * self.elements);
        }

        pub inline fn len(self: Self) Scalar {
            return @sqrt(self.sqrLen());
        }

        pub inline fn norm(self: Self) Self {
            const length = self.len();
            return if (length != 0) self.divScalar(length) else self;
        }

        // assmues unit quaternions
        pub inline fn angle(a: Self, b: Self) Scalar {
            const cos_theta = a.dot(b);
            return std.math.ascos(2 * cos_theta * cos_theta - 1);
        }

        pub inline fn negate(self: Self) Self {
            return self.mulScalar(-1.0);
        }

        pub inline fn addScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements + splat(s) };
        }

        pub inline fn subScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements - splat(s) };
        }

        pub inline fn mulScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements * splat(s) };
        }

        pub inline fn divScalar(a: Self, s: Scalar) Self {
            return .{ .elements = a.elements / splat(s) };
        }

        pub inline fn isInf(self: Self) bool {
            var is_inf = false;
            inline for (0..4) |i| {
                is_inf = std.math.isInf(self.elements[i]) or is_inf;
            }
            return is_inf;
        }

        pub inline fn isNan(self: Self) bool {
            var is_nan = false;
            inline for (0..4) |i| {
                is_nan = std.math.isNan(self.elements[i]) or is_nan;
            }
            return is_nan;
        }

        inline fn splat(scalar: Scalar) @Vector(4, Scalar) {
            return @splat(scalar);
        }
    };
}

test "mul" {
    const Quat = GenericQuat(f32);
    {
        const a = Quat.init(1, 0, 0, 0);
        const b = Quat.init(0, 1, 0, 0);
        const ab = a.mul(b);
        try testing.expect(ab.eql(Quat.init(0, 0, 1, 0)));
        try testing.expect(!ab.eqlApprox(Quat.init(0, 0, 1, 1), 0.0001));
    }
    {
        const a = Quat.init(2, 4, 8, 16);
        const b = a.inverse();
        const ab = a.mul(b);
        try testing.expectEqual(0, ab.x());
        try testing.expectEqual(0, ab.y());
        try testing.expectEqual(0, ab.z());
        try testing.expectEqual(1, ab.w());
    }
}

test "rotate" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);

    const a = Quat.fromEulerAngles(Vec3.splat(0.785398));
    const v = Vec3.init(0, 0, 1);

    const expect = Vec3.init(8.535533e-1, -1.4644669e-1, 5.000001e-1);
    try testing.expectEqual(expect, a.rotate(v));
}

test "eulerAngles" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);

    const angles = Vec3.init(0.78539807, 0.523599, 0.2);
    const a = Quat.fromEulerAngles(Vec3.init(0.785398, 0.523599, 0.2));

    try testing.expect(angles.eqlApprox(a.toEulerAngles(), 0.000001));
}

test "axis, rotateX" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);

    const a = Quat.fromAxis(std.math.pi / 2.0, Vec3.init(1, 0, 0));
    const b = Quat.identity().rotateX(std.math.pi / 2.0);

    try testing.expectEqual(a, b);
}

test "axis, rotateY" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);

    const a = Quat.fromAxis(std.math.pi / 3.0, Vec3.init(0, 1, 0));
    const b = Quat.identity().rotateY(std.math.pi / 3.0);

    try testing.expectEqual(a, b);
}

test "axis, rotateZ" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);

    const a = Quat.fromAxis(std.math.pi / 4.0, Vec3.init(0, 0, 1));
    const b = Quat.identity().rotateZ(std.math.pi / 4.0);

    try testing.expectEqual(b, a);
}

test "toMat3" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);
    const Mat3 = GenericMatrix(3, 3, f32);

    const a = Quat.fromAxis(std.math.pi / 4.0, Vec3.init(0, 0, 1));

    try testing.expectEqual(Mat3.init(
        Vec3.init(7.071067e-1, 7.071068e-1, 0e0),
        Vec3.init(-7.071068e-1, 7.071067e-1, 0e0),
        Vec3.init(0e0, 0e0, 1e0),
    ), a.toMat3());
}

test "rotation" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);
    const Vec4 = GenericVector(4, f32);

    const aa = Quat.fromEulerAngles(Vec3.splat(0.785398));
    const v1 = Vec3.init(0, 0, 1);
    const v2 = Vec4.init(0, 0, 1, 1);

    const expect = Vec3.init(8.535533e-1, -1.4644669e-1, 5.000001e-1);

    try testing.expectEqual(expect, aa.rotate(v1));
    try testing.expectEqual(expect, aa.toMat4().mul(v2).swizzle("xyz"));
}

test "fromMat4" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);
    const Mat4 = GenericMatrix(4, 4, f32);

    const mat = Mat4.fromEulerAngles(Vec3.init(0.78539795, -0.0872665, 0.34906596));
    const quat = Quat.fromMat4(mat);

    try testing.expect(mat.toEulerAngles().eqlApprox(quat.toEulerAngles(), 0.00001));
}

test "dot" {
    const Quat = GenericQuat(f32);

    const a = Quat.init(0.1, 0.2, 0.3, 0.5);
    const b = Quat.init(0.5, 0.7, 0.7, 0.4);

    try testing.expectEqual(6e-1, a.dot(b));
}

test "slerp" {
    const Quat = GenericQuat(f32);
    const Vec3 = GenericVector(3, f32);

    {
        const a = Quat.init(0.1, 0.2, 0.3, 0.5);
        const b = Quat.init(0.5, 0.7, 0.7, 0.4);

        try testing.expectEqual(Quat.init(3.354102e-1, 5.031153e-1, 5.59017e-1, 5.031153e-1), a.slerp(b, 0.5));
    }
    {
        const a = Quat.identity();
        const b = Quat.fromAxis(3.14159, Vec3.init(0, 1, 0));
        try testing.expectEqual(Quat.init(0e0, 7.0710635e-1, 0e0, 7.0710725e-1), Quat.slerp(a, b, 0.5));
    }

    {
        const a = Quat.identity();
        const b = Quat.fromAxis(3.14159, Vec3.init(0, 1, 0));
        try testing.expectEqual(b, Quat.slerp(a, b, 1));
    }
}

test "isNan isInf" {
    const Quat = GenericQuat(f32);

    const a = Quat.init(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    const b = Quat.init(std.math.nan(f32), std.math.nan(f32), std.math.nan(f32), std.math.nan(f32));

    try testing.expect(a.isInf());
    try testing.expect(b.isNan());
}
