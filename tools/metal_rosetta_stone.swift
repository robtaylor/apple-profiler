import Metal
import Foundation

// "Rosetta stone" trace: exercises Metal compute APIs in a known sequence
// so we can definitively map function indices to API calls.
//
// The sequence is designed so each API call appears at a unique position,
// making it trivial to map function stream index → Metal API.
//
// Usage: MTL_CAPTURE_ENABLED=1 ./rosetta [--variant N]
//
// Variants:
//   0 (default): Full sequence — baseline for all index mappings
//   1: dispatchThreads (non-uniform) instead of dispatchThreadgroups
//   2: setBytes with different data
//   3: Two encoders in one command buffer
//   4: memoryBarrier(resources:) instead of memoryBarrier(scope:)
//   5: Multiple command buffers

let variantArg = CommandLine.arguments.firstIndex(of: "--variant")
    .map { CommandLine.arguments[$0 + 1] }
let variant = Int(variantArg ?? "0") ?? 0

let outputDir = "/tmp/claude/barrier_test"
let outputPath = "\(outputDir)/rosetta_v\(variant).gputrace"

print("Rosetta variant \(variant) → \(outputPath)")

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device")
}

// Trivial kernel
let shaderSource = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* buf [[buffer(0)]],
                    uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0;
}
kernel void mul_two(device float* buf [[buffer(0)]],
                    uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] * 2.0;
}
"""

let library = try! device.makeLibrary(source: shaderSource, options: nil)
let addFn = library.makeFunction(name: "add_one")!
let mulFn = library.makeFunction(name: "mul_two")!
let addPipeline = try! device.makeComputePipelineState(function: addFn)
let mulPipeline = try! device.makeComputePipelineState(function: mulFn)

let count = 256
let bufA = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                              options: .storageModeShared)!
let bufB = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                              options: .storageModeShared)!

// Init
let ptrA = bufA.contents().bindMemory(to: Float.self, capacity: count)
let ptrB = bufB.contents().bindMemory(to: Float.self, capacity: count)
for i in 0..<count { ptrA[i] = Float(i); ptrB[i] = Float(i) * 0.5 }

// Capture setup
let captureManager = MTLCaptureManager.shared()
let captureDescriptor = MTLCaptureDescriptor()
captureDescriptor.captureObject = device
captureDescriptor.destination = .gpuTraceDocument
captureDescriptor.outputURL = URL(fileURLWithPath: outputPath)
try? FileManager.default.removeItem(atPath: outputPath)

do {
    try captureManager.startCapture(with: captureDescriptor)
} catch {
    fatalError("Failed to start capture: \(error)")
}

let queue = device.makeCommandQueue()!

switch variant {
case 0:
    // VARIANT 0: Baseline — full API exercise
    // Sequence:
    //   commandBuffer → computeCommandEncoder →
    //   setComputePipelineState(add) → setBuffer(bufA, idx=0) →
    //   dispatchThreadgroups → memoryBarrier(scope: .buffers) →
    //   setComputePipelineState(mul) → setBuffer(bufB, idx=0) →
    //   dispatchThreadgroups →
    //   endEncoding → commit → waitUntilCompleted
    print("  commandBuffer")
    let cb = queue.makeCommandBuffer()!
    print("  computeCommandEncoder")
    let enc = cb.makeComputeCommandEncoder()!

    print("  setComputePipelineState (add_one)")
    enc.setComputePipelineState(addPipeline)
    print("  setBuffer (bufA at index 0)")
    enc.setBuffer(bufA, offset: 0, index: 0)
    print("  dispatchThreadgroups {1,1,1} threads {256,1,1}")
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))

    print("  memoryBarrier(scope: .buffers)")
    enc.memoryBarrier(scope: .buffers)

    print("  setComputePipelineState (mul_two)")
    enc.setComputePipelineState(mulPipeline)
    print("  setBuffer (bufB at index 0)")
    enc.setBuffer(bufB, offset: 0, index: 0)
    print("  dispatchThreadgroups {1,1,1} threads {256,1,1}")
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))

    print("  endEncoding")
    enc.endEncoding()
    print("  commit")
    cb.commit()
    print("  waitUntilCompleted")
    cb.waitUntilCompleted()

case 1:
    // VARIANT 1: Use dispatchThreads (non-uniform) instead of dispatchThreadgroups
    print("  Using dispatchThreads (non-uniform threadgroups)")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)

    // dispatchThreads:threadsPerThreadgroup: — the non-uniform variant
    enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: min(count, addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

case 2:
    // VARIANT 2: setBytes with inline data (not buffer pointer)
    print("  Using setBytes with inline data")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)

    // setBytes — push small inline data
    var multiplier: Float = 3.14
    enc.setBytes(&multiplier, length: MemoryLayout<Float>.stride, index: 1)

    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

case 3:
    // VARIANT 3: Two separate encoders in one command buffer
    print("  Two encoders in one command buffer")
    let cb = queue.makeCommandBuffer()!

    // First encoder
    let enc1 = cb.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(addPipeline)
    enc1.setBuffer(bufA, offset: 0, index: 0)
    enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc1.endEncoding()

    // Second encoder
    let enc2 = cb.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(mulPipeline)
    enc2.setBuffer(bufB, offset: 0, index: 0)
    enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc2.endEncoding()

    cb.commit()
    cb.waitUntilCompleted()

case 4:
    // VARIANT 4: memoryBarrier(resources:) instead of memoryBarrier(scope:)
    print("  Using memoryBarrier(resources:)")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))

    // Resource-specific barrier
    enc.memoryBarrier(resources: [bufA])

    enc.setComputePipelineState(mulPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

case 5:
    // VARIANT 5: Two separate command buffers
    print("  Two command buffers")
    let cb1 = queue.makeCommandBuffer()!
    let enc1 = cb1.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(addPipeline)
    enc1.setBuffer(bufA, offset: 0, index: 0)
    enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc1.endEncoding()
    cb1.commit()
    cb1.waitUntilCompleted()

    let cb2 = queue.makeCommandBuffer()!
    let enc2 = cb2.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(mulPipeline)
    enc2.setBuffer(bufB, offset: 0, index: 0)
    enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc2.endEncoding()
    cb2.commit()
    cb2.waitUntilCompleted()

case 6:
    // VARIANT 6: Blit command encoder (test if endEncoding has different idx)
    print("  Blit encoder (copy buffer)")
    let cb = queue.makeCommandBuffer()!
    let blitEnc = cb.makeBlitCommandEncoder()!
    blitEnc.copy(from: bufA, sourceOffset: 0, to: bufB, destinationOffset: 0, size: count * MemoryLayout<Float>.stride)
    blitEnc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

case 7:
    // VARIANT 7: makeBuffer(bytes:length:options:) to find -16314
    print("  makeBuffer(bytes:) instead of makeBuffer(length:)")
    var data = [Float](repeating: 1.0, count: count)
    let bufC = device.makeBuffer(bytes: &data,
                                  length: count * MemoryLayout<Float>.stride,
                                  options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addPipeline)
    enc.setBuffer(bufC, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

case 8:
    // VARIANT 8: addCompletedHandler to find -15990
    print("  addCompletedHandler")
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()
    let sem = DispatchSemaphore(value: 0)
    cb.addCompletedHandler { _ in
        print("    Completed handler fired")
        sem.signal()
    }
    cb.commit()
    sem.wait()

case 9:
    // VARIANT 9: MTLComputePipelineDescriptor to find -15996
    print("  makeComputePipelineState(descriptor:)")
    let descriptor = MTLComputePipelineDescriptor()
    descriptor.computeFunction = addFn
    descriptor.label = "test_descriptor_pipeline"
    let descPipeline = try! device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(descPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

case 10:
    // VARIANT 10: MTLSharedEvent to find -15422
    print("  MTLSharedEvent signaling")
    let event = device.makeSharedEvent()!
    let listener = MTLSharedEventListener()

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()

    // Signal event after dispatch completes
    cb.encodeSignalEvent(event, value: 1)
    cb.commit()

    // Wait on CPU side for the event
    let sem = DispatchSemaphore(value: 0)
    event.notify(listener, atValue: 1) { _, _ in
        print("    Event signaled")
        sem.signal()
    }
    sem.wait()

case 11:
    // VARIANT 11: Pre-compiled metallib loaded with newLibraryWithURL
    // to test if -16290 (newFunctionWithName) appears separately
    print("  Load metallib from file")
    // First compile to a metallib file
    let metalSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void add_one(device float* buf [[buffer(0)]],
                        uint tid [[thread_position_in_grid]]) {
        buf[tid] = buf[tid] + 1.0;
    }
    """
    let tempLib = try! device.makeLibrary(source: metalSource, options: nil)
    // We can't easily save a metallib at runtime, so instead just call
    // makeFunction(name:) explicitly on the source-compiled library
    // to see what index it produces
    let fn = tempLib.makeFunction(name: "add_one")!
    let pl = try! device.makeComputePipelineState(function: fn)

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pl)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

case 12:
    // VARIANT 12: setPurgeableState to find -16371
    print("  setPurgeableState on buffer")
    let _ = bufA.setPurgeableState(.nonVolatile)
    let _ = bufA.setPurgeableState(.volatile)
    let _ = bufA.setPurgeableState(.nonVolatile)

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(addPipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

default:
    fatalError("Unknown variant \(variant)")
}

captureManager.stopCapture()
print("Done. Trace at: \(outputPath)")
