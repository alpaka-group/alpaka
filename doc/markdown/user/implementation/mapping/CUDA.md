[:arrow_up: Up](../Mapping.md)

CUDA GPUs
=========

Mapping the abstraction to GPUs supporting *CUDA* is straightforward because the hierarchy levels are identical up to the element level.
So blocks of warps of threads will be mapped directly to their *CUDA* equivalent.

The element level will be supported through an additional run-time variable containing the extent of elements per thread.
This variable can be accessed by all threads and should optimally be placed in constant device memory for fast access.

Porting CUDA to *alpaka*
------------------------

Nearly all CUDA functionality can be directly mapped to alpaka function calls.
A major difference is that CUDA requires the block and grid sizes to be given in (x, y, z) order. Alpaka uses the mathematical C/C++ array indexing scheme [z][y][x]. Dimension 0 in this case is z, dimensions 2 is x.

Furthermore alpaka does not require the indices and extents to be 3-dimensional.
The accelerators are templatized on and support arbitrary dimensionality.
NOTE: Currently the CUDA implementation is restricted to a maximum of 3 dimensions!

NOTE: The CUDA-accelerator back-end can change the current CUDA device and will NOT set the device back to the one prior to the invocation of the alpaka function!

The following tables list the functions available in the [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules) and their equivalent alpaka functions:

*Device Management*

|CUDA|alpaka|
|---|---|
|cudaChooseDevice|-|
|cudaDeviceGetByPCIBusId|-|
|cudaDeviceGetAttribute|-|
|cudaDeviceGetCacheConfig|-|
|cudaDeviceGetLimit|-|
|cudaDeviceGetPCIBusId|-|
|cudaDeviceGetSharedMemConfig|-|
|cudaDeviceGetStreamPriorityRange|-|
|cudaDeviceReset|alpaka::dev::reset(device)|
|cudaDeviceSetCacheConfig|-|
|cudaDeviceSetLimit|-|
|cudaDeviceSetSharedMemConfig|-|
|cudaDeviceSynchronize|void alpaka::wait::wait(device)|
|cudaGetDevice|n/a (no current device)|
|cudaGetDeviceCount|std::size_t alpaka::dev::DevMan< TAcc >::getDeviceCount()|
|cudaGetDeviceProperties|<ul><li>alpaka::dev::DevProps alpaka::dev::getProps(device)</li><li>alpaka::acc::getAccDevProps(acc)</li></ul> *NOTE: Only some properties available*|
|cudaIpcCloseMemHandle|-|
|cudaIpcGetEventHandle|-|
|cudaIpcGetMemHandle|-|
|cudaIpcOpenEventHandle|-|
|cudaIpcOpenMemHandle|-|
|cudaSetDevice|n/a (no current device)|
|cudaSetDeviceFlags|-|
|cudaSetValidDevices|-|

*Stream Management*

|CUDA|alpaka|
|---|---|
|cudaStreamAddCallback|-|
|cudaStreamAttachMemAsync|-|
|cudaStreamCreate|alpaka::stream::[StreamType] stream(device);|
|cudaStreamCreateWithFlags|-|
|cudaStreamCreateWithPriority|-|
|cudaStreamDestroy|n/a (Destructor)|
|cudaStreamGetFlags|-|
|cudaStreamGetPriority|-|
|cudaStreamQuery|bool alpaka::stream::test(stream)|
|cudaStreamSynchronize|void alpaka::wait::wait(stream)|
|cudaStreamWaitEvent|void alpaka::wait::wait(stream, event)|

*Event Management*

|CUDA|alpaka|
|---|---|
|cudaEventCreate|alpaka::event::Event< TAcc > event(device);|
|cudaEventCreateWithFlags|-|
|cudaEventDestroy|n/a (Destructor)|
|cudaEventElapsedTime|-|
|cudaEventQuery|bool alpaka::event::test(event)|
|cudaEventRecord|void alpaka::stream::enqueue(stream, event)|
|cudaEventSynchronize|void alpaka::wait::wait(event)|

*Memory Management*

|CUDA|alpaka|
|---|---|
|cudaArrayGetInfo|-|
|cudaFree|n/a (automatic memory management with reference counted memory handles)|
|cudaFreeArray|-|
|cudaFreeHost|n/a|
|cudaFreeMipmappedArray|-|
|cudaGetMipmappedArrayLevel|-|
|cudaGetSymbolAddress|-|
|cudaGetSymbolSize|-|
|cudaHostAlloc|n/a|
|cudaHostGetDevicePointer|-|
|cudaHostGetFlags|-|
|cudaHostRegister|-|
|cudaHostUnregister|-|
|cudaMalloc|alpaka::mem::buf::alloc<TElement>(device, extents1D)|
|cudaMalloc3D|alpaka::mem::buf::alloc<TElement>(device, extents3D)|
|cudaMalloc3DArray|-|
|cudaMallocArray|-|
|cudaMallocHost|alpaka::mem::buf::alloc<TElement>(device, extents) *1D, 2D, 3D suppoorted!*|
|cudaMallocManaged|TODO|
|cudaMallocMipmappedArray|-|
|cudaMallocPitch|alpaka::mem::alloc<TElement>(device, extents2D)|
|cudaMemGetInfo|<ul><li>alpaka::dev::getMemBytes</li><li>alpaka::dev::getFreeMemBytes</li><ul>|
|cudaMemcpy|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D)|
|cudaMemcpy2D|alpaka::mem::view::copy(memBufDst, memBufSrc, extents2D)|
|cudaMemcpy2DArrayToArray|-|
|cudaMemcpy2DAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents2D, stream)|
|cudaMemcpy2DFromArray|-|
|cudaMemcpy2DFromArrayAsync|-|
|cudaMemcpy2DToArray|-|
|cudaMemcpy2DToArrayAsync|-|
|cudaMemcpy3D|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D)|
|cudaMemcpy3DAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D, stream)|
|cudaMemcpy3DPeer|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D)|
|cudaMemcpy3DPeerAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D, stream)|
|cudaMemcpyArrayToArray|-|
|cudaMemcpyAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D, stream)|
|cudaMemcpyFromArray|-|
|cudaMemcpyFromArrayAsync|-|
|cudaMemcpyFromSymbol|-|
|cudaMemcpyFromSymbolAsync|-|
|cudaMemcpyPeer|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D)|
|cudaMemcpyPeerAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D, stream)|
|cudaMemcpyToArray|-|
|cudaMemcpyToArrayAsync|-|
|cudaMemcpyToSymbol|-|
|cudaMemcpyToSymbolAsync|-|
|cudaMemset|alpaka::mem::view::set(memBufDst, byte, extents1D)|
|cudaMemset2D|alpaka::mem::view::set(memBufDst, byte, extents2D)|
|cudaMemset2DAsync|alpaka::mem::view::set(memBufDst, byte, extents2D, stream)|
|cudaMemset3D|alpaka::mem::view::set(memBufDst, byte, extents3D)|
|cudaMemset3DAsync|alpaka::mem::view::set(memBufDst, byte, extents3D, stream)|
|cudaMemsetAsync|alpaka::mem::view::set(memBufDst, byte, extents1D, stream)|
|make_cudaExtent|-|
|make_cudaPitchedPtr|-|

*Execution Control*

|CUDA|alpaka|
|---|---|
|cudaConfigureCall|<ul><li>alpaka::stream::enqueue(stream, kernel, params...)</li><li>alpaka::kernel::BlockSharedExternMemSizeBytes< TKernel< TAcc > >::getBlockSharedExternMemSizeBytes<...>(...)</li></ul>|
|cudaFuncGetAttributes|-|
|cudaFuncSetCacheConfig|-|
|cudaFuncSetSharedMemConfig|-|
|cudaLaunch|alpaka::stream::enqueue(stream, kernel, params...)|
|cudaSetDoubleForDevice|n/a (alpaka assumes double support)|
|cudaSetDoubleForHost|n/a (alpaka assumes double support)|
|cudaSetupArgument|alpaka::stream::enqueue(stream, kernel, params...)|

*Occupancy*

|CUDA|alpaka|
|---|---|
|cudaOccupancyMaxActiveBlocksPerMultiprocessor|-|


*Unified Addressing*

|CUDA|alpaka|
|---|---|
|cudaPointerGetAttributes|alpaka::mem::view::getPtrDev(view, device)|

*Peer Device Memory Access*

|CUDA|alpaka|
|---|---|
|cudaDeviceCanAccessPeer|-|
|cudaDeviceDisablePeerAccess|-|
|cudaDeviceEnablePeerAccess|-|

**OpenGL, DirectX, VDPAU, Graphics Interoperability**

*not available*

**Texture/Surface Reference/Object Management**

*not available*

**Version Management**

*not available*