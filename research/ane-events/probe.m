// Standalone private-API research. Never link this into TurboSpark.
// Interfaces inspected at phhusson/llama.cpp abcc7f8fa8920f50bc321ee7e7dcfa31b133a6a0.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <unistd.h>
#include <math.h>

@interface NSObject (ANEProbe)
+ (id)modelWithMILText:(NSData *)mil weights:(NSDictionary *)weights optionsPlist:(id)options;
+ (id)inMemoryModelWithDescriptor:(id)descriptor;
- (NSString *)hexStringIdentifier;
- (BOOL)compileWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)loadWithQoS:(unsigned int)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)unloadWithQoS:(unsigned int)qos error:(NSError **)error;
- (BOOL)evaluateWithQoS:(unsigned int)qos options:(NSDictionary *)options request:(id)request error:(NSError **)error;
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices outputs:(NSArray *)outputs outputIndices:(NSArray *)outputIndices weightsBuffer:(id)weights perfStats:(id)stats procedureIndex:(NSNumber *)index sharedEvents:(id)events;
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(uint32_t)index eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
- (id)IOSurfaceSharedEvent;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError *error))handler;
@end

static void require(BOOL condition, NSString *reason) {
    if (!condition) { fprintf(stderr, "%s\n", reason.UTF8String); exit(3); }
}

static IOSurfaceRef plane(void) {
    size_t bytes = (size_t)getpagesize();
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    require(surface != NULL, @"IOSurfaceCreate failed");
    return surface;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        (void)argc; (void)argv;
        void *framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW|RTLD_LOCAL);
        for (NSString *name in @[@"_ANEInMemoryModel", @"_ANEInMemoryModelDescriptor", @"_ANEIOSurfaceObject", @"_ANERequest", @"_ANESharedEvents", @"_ANESharedWaitEvent", @"_ANESharedSignalEvent"]) {
            if (!framework || !NSClassFromString(name)) { fprintf(stderr, "unavailable: %s\n", name.UTF8String); return 77; }
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> producerQueue = [device newCommandQueue];
        id<MTLCommandQueue> consumerQueue = [device newCommandQueue];
        id<MTLSharedEvent> ready = [device newSharedEvent], done = [device newSharedEvent];
        id<MTLSharedEvent> producerGate = [device newSharedEvent], aneGate = [device newSharedEvent];
        if (![(id)ready respondsToSelector:@selector(IOSurfaceSharedEvent)]) { fprintf(stderr, "IOSurfaceSharedEvent unavailable\n"); return 77; }
        NSString *mil = @"program(1.3)\n"
            "[buildInfo = dict<string,string>({{\"coremlc-component-MIL\",\"3510.2.1\"},{\"coremlc-version\",\"3505.4.1\"},{\"coremltools-component-milinternal\",\"\"},{\"coremltools-version\",\"9.0\"}})]\n"
            "{ func main<ios18>(tensor<fp16,[1,1,32,64]> a, tensor<fp16,[1,1,64,64]> w) {\n"
            "bool no = const()[name=string(\"no\"),val=bool(false)];\n"
            "bool yes = const()[name=string(\"yes\"),val=bool(true)];\n"
            "tensor<fp16,[1,1,64,32]> y = matmul(x=w,y=a,transpose_x=no,transpose_y=yes)[name=string(\"y\")];\n"
            "} -> (y); }\n";
        NSData *data = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor = [NSClassFromString(@"_ANEInMemoryModelDescriptor") modelWithMILText:data weights:@{@"@model_path/weights/weight.bin": @{@"offset":@0,@"data":[NSData dataWithBytes:"\0\0\0\0" length:4]}} optionsPlist:nil];
        require(descriptor != nil, @"MIL descriptor failed");
        id model = [NSClassFromString(@"_ANEInMemoryModel") inMemoryModelWithDescriptor:descriptor];
        require(model != nil, @"in-memory model failed");
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:[model hexStringIdentifier]];
        NSError *error = nil;
        require([[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:&error], error.description);
        require([data writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] atomically:YES], @"MIL write failed");
        require([model compileWithQoS:21 options:@{} error:&error], error.description ?: @"compile failed");
        require([model loadWithQoS:21 options:@{} error:&error], error.description ?: @"load failed");
        IOSurfaceRef a = plane(), w = plane(), y = plane();
        _Float16 *weight = IOSurfaceGetBaseAddress(w);
        memset(weight, 0, IOSurfaceGetAllocSize(w));
        for (int d=0; d<64; d++) weight[d*64+d] = 1;
        id<MTLBuffer> input = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(a) length:IOSurfaceGetAllocSize(a) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> output = [device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(y) length:IOSurfaceGetAllocSize(y) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> source = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        id<MTLBuffer> result = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        require(input && output && source && result, @"Metal allocation failed");
        for (uint64_t sequence=1; sequence<=6; sequence++) {
            @autoreleasepool {
                // Change sentinels on every invocation so stale output cannot pass.
                for (int i=0; i<2048; i++) ((_Float16 *)source.contents)[i] = (_Float16)(sequence + (i%7)*0.125f);
                for (int i=0; i<2048; i++) ((_Float16 *)IOSurfaceGetBaseAddress(y))[i] = -99;
                BOOL producerStall = (sequence % 2) == 1;
                if (producerStall) aneGate.signaledValue = sequence;
                else producerGate.signaledValue = sequence;
                id waitReady = [NSClassFromString(@"_ANESharedWaitEvent") waitEventWithValue:sequence sharedEvent:[(id)ready IOSurfaceSharedEvent]];
                id waitANE = [NSClassFromString(@"_ANESharedWaitEvent") waitEventWithValue:sequence sharedEvent:[(id)aneGate IOSurfaceSharedEvent]];
                id signal = [NSClassFromString(@"_ANESharedSignalEvent") signalEventWithValue:sequence symbolIndex:0 eventType:0 sharedEvent:[(id)done IOSurfaceSharedEvent]];
                id events = [NSClassFromString(@"_ANESharedEvents") sharedEventsWithSignalEvents:@[signal] waitEvents:@[waitReady, waitANE]];
                id request = [NSClassFromString(@"_ANERequest") requestWithInputs:@[[NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:a], [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:w]] inputIndices:@[@0,@1] outputs:@[[NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:y]] outputIndices:@[@0] weightsBuffer:nil perfStats:nil procedureIndex:@0 sharedEvents:events];
                require(request != nil, @"request construction failed");
                dispatch_semaphore_t completed = dispatch_semaphore_create(0);
                __block BOOL success = NO;
                [request setCompletionHandler:^(BOOL ok, NSError *failure) { success = ok; if (failure) fprintf(stderr,"%s\n",failure.description.UTF8String); dispatch_semaphore_signal(completed); }];
                id<MTLCommandBuffer> producer = [producerQueue commandBuffer];
                [producer encodeWaitForEvent:producerGate value:sequence];
                id<MTLBlitCommandEncoder> blit = [producer blitCommandEncoder];
                [blit copyFromBuffer:source sourceOffset:0 toBuffer:input destinationOffset:0 size:4096]; [blit endEncoding];
                [producer encodeSignalEvent:ready value:sequence]; [producer commit];
                id<MTLCommandBuffer> consumer = [consumerQueue commandBuffer];
                [consumer encodeWaitForEvent:done value:sequence];
                blit = [consumer blitCommandEncoder];
                [blit copyFromBuffer:output sourceOffset:0 toBuffer:result destinationOffset:0 size:4096]; [blit endEncoding]; [consumer commit];
                // Submission itself may block on some OS versions. A worker
                // lets the probe detect that without deadlocking the gate test.
                dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^{
                    NSError *failure = nil;
                    BOOL ok = [model evaluateWithQoS:21 options:@{} request:request error:&failure];
                    if (!ok) { fprintf(stderr,"submission failed: %s\n",failure.description.UTF8String); exit(3); }
                });
                usleep(100000);
                require(done.signaledValue < sequence && consumer.status != MTLCommandBufferStatusCompleted,
                    producerStall ? @"consumer crossed stalled producer" : @"consumer crossed delayed ANE dependency");
                require(((_Float16 *)IOSurfaceGetBaseAddress(y))[0] == -99, @"ANE wrote before dependency release");
                if (producerStall) producerGate.signaledValue = sequence;
                else aneGate.signaledValue = sequence;
                [producer waitUntilCompleted];
                require(producer.status == MTLCommandBufferStatusCompleted, @"producer failed");
                require(dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 10*NSEC_PER_SEC)) == 0, @"ANE completion timed out; exiting process without faking event");
                require(success, @"ANE execution failed");
                [consumer waitUntilCompleted];
                require(consumer.status == MTLCommandBufferStatusCompleted, @"consumer failed");
                for (int d=0; d<64; d++) for (int t=0; t<32; t++) {
                    require(((_Float16 *)result.contents)[d*32+t] == ((_Float16 *)source.contents)[t*64+d], @"stale or incorrect ANE output");
                }
                printf("{\"iteration\":%llu,\"producer_stall\":%s,\"ane_stall\":%s,\"sentinel_parity\":true}\n", sequence, producerStall ? "true" : "false", producerStall ? "false" : "true");
            }
        }
        require([model unloadWithQoS:21 error:&error], error.description ?: @"unload failed");
        // Wrappers leave scope before releasing their no-copy backing.
        input=nil; output=nil;
        CFRelease(a); CFRelease(w); CFRelease(y);
        [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
    }
    return 0;
}
