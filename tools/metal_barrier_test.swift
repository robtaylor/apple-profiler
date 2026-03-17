import Metal
import Foundation

// Minimal Metal compute program to test barrier hypothesis.
// Variant A (--barrier): dispatch → memoryBarrierWithScope → dispatch
// Variant B (--no-barrier): dispatch → dispatch (no barrier)
//
// Captures a .gputrace via MTLCaptureManager so we can diff the function streams.

let useBarrier = CommandLine.arguments.contains("--barrier")
let outputPath = CommandLine.arguments.contains("--output")
    ? CommandLine.arguments[CommandLine.arguments.firstIndex(of: "--output")! + 1]
    : "/tmp/claude/barrier_test/\(useBarrier ? "with_barrier" : "no_barrier").gputrace"

print("Mode: \(useBarrier ? "WITH barrier" : "NO barrier")")
print("Output: \(outputPath)")

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device")
}

// Trivial kernel: buf[tid] = buf[tid] + 1
let shaderSource = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* buf [[buffer(0)]],
                    uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0;
}
"""

let library = try! device.makeLibrary(source: shaderSource, options: nil)
let function = library.makeFunction(name: "add_one")!
let pipeline = try! device.makeComputePipelineState(function: function)

// Create a small buffer
let count = 256
let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                                options: .storageModeShared)!

// Initialize buffer
let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
for i in 0..<count { ptr[i] = Float(i) }

// Set up GPU capture
let captureManager = MTLCaptureManager.shared()
let captureDescriptor = MTLCaptureDescriptor()
captureDescriptor.captureObject = device
captureDescriptor.destination = .gpuTraceDocument
captureDescriptor.outputURL = URL(fileURLWithPath: outputPath)

// Remove existing trace if present
try? FileManager.default.removeItem(atPath: outputPath)

do {
    try captureManager.startCapture(with: captureDescriptor)
} catch {
    fatalError("Failed to start capture: \(error)")
}

let queue = device.makeCommandQueue()!
let cmdBuf = queue.makeCommandBuffer()!
let encoder = cmdBuf.makeComputeCommandEncoder()!

// First dispatch
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(buffer, offset: 0, index: 0)
encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))

if useBarrier {
    // Explicit memory barrier between dispatches
    encoder.memoryBarrier(scope: .buffers)
}

// Second dispatch (reads what the first wrote)
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(buffer, offset: 0, index: 0)
encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))

encoder.endEncoding()
cmdBuf.commit()
cmdBuf.waitUntilCompleted()

captureManager.stopCapture()

// Verify result: each element should be original + 2
let result = buffer.contents().bindMemory(to: Float.self, capacity: count)
let expected = useBarrier ? "correct (barrier ensures ordering)" : "possibly correct (no barrier)"
print("buf[0] = \(result[0]) (expected 2.0) — \(expected)")
print("buf[255] = \(result[255]) (expected 257.0)")
print("Trace saved to: \(outputPath)")
