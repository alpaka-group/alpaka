#include <alpaka/alpaka.hpp>

// ===== Begin: main first part =====
// This code section needs to be global, that I can hide the code in the documentation
// NOLINTBEGIN(bugprone-throwing-static-initialization)
auto devSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
auto devSelector = alpaka::onHost::makeDeviceSelector(devSpec);
auto numComputeDevs = devSelector.getDeviceCount();

alpaka::onHost::Device dev = devSelector.makeDevice(0);
alpaka::onHost::Queue queue = dev.makeQueue();

alpaka::Vec extents{111};

alpaka::onHost::SharedBuffer in = alpaka::onHost::alloc<int>(dev, extents);
alpaka::onHost::SharedBuffer out = alpaka::onHost::alloc<int>(dev, extents);

alpaka::onHost::concepts::FrameSpec auto frameSpec
    = alpaka::onHost::getFrameSpec(dev, alpaka::exec::anyExecutor, extents);

// NOLINTEND(bugprone-throwing-static-initialization)
// ===== End: main first part =====


// BEGIN-DATASTORAGE-writeableDatasource

// same interface like alpaka::onAcc::SimdAlgo.concurrent()
ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
    auto const& acc,
    auto&& func,
    alpaka::concepts::IDataSource auto&& data0,
    alpaka::concepts::IDataSource auto&&... dataN)
{
    auto simdGrid = alpaka::onAcc::SimdAlgo{alpaka::onAcc::worker::threadsInGrid};
    simdGrid.concurrent(acc, data0.getExtents(), func, data0, dataN...);
}

// user defined
struct SimdCopyOp
{
    constexpr void operator()(auto const&, auto const a, auto c) const
    {
        c = a.load();
    }
};

// user defined
struct Kernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto const& func,
        alpaka::concepts::IDataSource auto const& in,
        alpaka::concepts::IMdSpan auto out) const
    {
        concurrent(acc, func, in, out);
    }
};

int main()
{
    // ...

    queue.enqueue(frameSpec, alpaka::KernelBundle{Kernel{}, SimdCopyOp{}, in, out});

    return 0;
}

// END-DATASTORAGE-writeableDatasource
