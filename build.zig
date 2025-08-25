const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const package_moudle = b.addModule("zig_matrix", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const build_test = b.addTest(.{
        .root_module = package_moudle,
        // vector operations tests break without this
        .use_llvm = true,
    });
    const run_tests = b.addRunArtifact(build_test);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
