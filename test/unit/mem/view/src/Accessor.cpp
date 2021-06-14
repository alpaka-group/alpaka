/* Copyright 2021 Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/mem/view/Accessor.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewStdArray.hpp>
#include <alpaka/mem/view/ViewStdVector.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>

#include <catch2/catch.hpp>

TEST_CASE("IsView", "[accessor]")
{
    using alpaka::traits::internal::IsView;

    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using Dev = alpaka::Dev<Acc>;

    // buffer
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto buffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});
    STATIC_REQUIRE(IsView<decltype(buffer)>::value);

    // views
    STATIC_REQUIRE(IsView<alpaka::ViewPlainPtr<Dev, int, Dim, Size>>::value);
    STATIC_REQUIRE(IsView<std::array<int, 42>>::value);
    STATIC_REQUIRE(IsView<std::vector<int>>::value);
    STATIC_REQUIRE(IsView<alpaka::ViewSubView<Dev, int, Dim, Size>>::value);

    // accessor
    auto accessor = alpaka::access(buffer);
    STATIC_REQUIRE(!IsView<decltype(accessor)>::value);
}

namespace
{
    constexpr auto N = 1024;

    struct WriteKernelTemplate
    {
        template<typename TAcc, typename TAccessor>
        ALPAKA_FN_ACC void operator()(TAcc const&, TAccessor data) const
        {
            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::DimInt<1>, alpaka::Idx<TAcc>>{alpaka::Idx<TAcc>{3}}] = 3.0f;
        }
    };

    struct WriteKernelExplicit
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpaka::Accessor<TMemoryHandle, float, TIdx, 1, alpaka::WriteAccess> const data) const
        {
            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}] = 3.0f;
        }
    };

    struct ReadKernelTemplate
    {
        template<typename TAcc, typename TAccessor>
        ALPAKA_FN_ACC void operator()(TAcc const&, TAccessor data) const
        {
            float const v1 = data[1];
            float const v2 = data(2);
            float const v3 = data[alpaka::Vec<alpaka::DimInt<1>, alpaka::Idx<TAcc>>{alpaka::Idx<TAcc>{3}}];
            (void) v1;
            (void) v2;
            (void) v3;
        }
    };

    struct ReadKernelExplicit
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpaka::Accessor<TMemoryHandle, float, TIdx, 1, alpaka::ReadAccess> const data) const
        {
            float const v1 = data[1];
            float const v2 = data(2);
            float const v3 = data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}];
            (void) v1;
            (void) v2;
            (void) v3;
        }
    };

    struct ReadWriteKernelExplicit
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpaka::Accessor<TMemoryHandle, float, TIdx, 1, alpaka::ReadWriteAccess> const data) const
        {
            float const v1 = data[1];
            float const v2 = data(2);
            float const v3 = data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}];
            (void) v1;
            (void) v2;
            (void) v3;

            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::DimInt<1>, TIdx>{TIdx{3}}] = 3.0f;
        }
    };
} // namespace

TEST_CASE("readWrite", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto buffer = alpaka::allocBuf<float, Size>(devAcc, Size{N});
    auto const workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};

    alpaka::exec<Acc>(queue, workdiv, WriteKernelTemplate{}, alpaka::writeAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, WriteKernelExplicit{}, alpaka::writeAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadKernelTemplate{}, alpaka::readAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadKernelExplicit{}, alpaka::readAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadWriteKernelExplicit{}, alpaka::access(buffer));
}

namespace
{
    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx, std::size_t TDim>
    struct AccessorWithProjection;

    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx>
    struct AccessorWithProjection<TProjection, TMemoryHandle, TElem, TBufferIdx, 1>
    {
        ALPAKA_FN_ACC auto operator[](alpaka::Vec<alpaka::DimInt<1>, TBufferIdx> i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator[](TBufferIdx i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx i) const -> TElem
        {
            return TProjection{}(accessor(i));
        }

        alpaka::Accessor<TMemoryHandle, TElem, TBufferIdx, 1, alpaka::ReadAccess> accessor;
    };

    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx>
    struct AccessorWithProjection<TProjection, TMemoryHandle, TElem, TBufferIdx, 2>
    {
        ALPAKA_FN_ACC auto operator[](alpaka::Vec<alpaka::DimInt<2>, TBufferIdx> i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return TProjection{}(accessor(y, x));
        }

        alpaka::Accessor<TMemoryHandle, TElem, TBufferIdx, 2, alpaka::ReadAccess> accessor;
    };

    template<typename TProjection, typename TMemoryHandle, typename TElem, typename TBufferIdx>
    struct AccessorWithProjection<TProjection, TMemoryHandle, TElem, TBufferIdx, 3>
    {
        ALPAKA_FN_ACC auto operator[](alpaka::Vec<alpaka::DimInt<3>, TBufferIdx> i) const -> TElem
        {
            return TProjection{}(accessor[i]);
        }

        ALPAKA_FN_ACC auto operator()(TBufferIdx z, TBufferIdx y, TBufferIdx x) const -> TElem
        {
            return TProjection{}(accessor(z, y, x));
        }

        alpaka::Accessor<TMemoryHandle, TElem, TBufferIdx, 3, alpaka::ReadAccess> accessor;
    };

    struct DoubleValue
    {
        ALPAKA_FN_ACC auto operator()(int i) const
        {
            return i * 2;
        }
    };

    struct CopyKernel
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpaka::Accessor<TMemoryHandle, int, TIdx, 1, alpaka::ReadAccess> const src,
            alpaka::Accessor<TMemoryHandle, int, TIdx, 1, alpaka::WriteAccess> const dst) const
        {
            auto const projSrc = AccessorWithProjection<DoubleValue, TMemoryHandle, int, TIdx, 1>{src};
            dst[0] = projSrc[0];
        }
    };
} // namespace

TEST_CASE("projection", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};

    auto srcBuffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});
    auto dstBuffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});

    std::array<int, 1> host{{42}};
    alpaka::memcpy(queue, srcBuffer, host, 1);

    auto const workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};
    alpaka::exec<Acc>(queue, workdiv, CopyKernel{}, alpaka::readAccess(srcBuffer), alpaka::writeAccess(dstBuffer));

    alpaka::memcpy(queue, host, dstBuffer, 1);

    REQUIRE(host[0] == 84);
}

TEST_CASE("constraining", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto buffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});
    using MemoryHandle = alpaka::MemoryHandle<decltype(alpaka::access(buffer))>;

    alpaka::Accessor<
        MemoryHandle,
        int,
        Size,
        1,
        std::tuple<alpaka::ReadAccess, alpaka::WriteAccess, alpaka::ReadWriteAccess>>
        acc = alpaka::accessWith<alpaka::ReadAccess, alpaka::WriteAccess, alpaka::ReadWriteAccess>(buffer);

    // constraining from multi-tag to single-tag
    alpaka::Accessor<MemoryHandle, int, Size, 1, alpaka::ReadAccess> readAcc = alpaka::readAccess(acc);
    alpaka::Accessor<MemoryHandle, int, Size, 1, alpaka::WriteAccess> writeAcc = alpaka::writeAccess(acc);
    alpaka::Accessor<MemoryHandle, int, Size, 1, alpaka::ReadWriteAccess> readWriteAcc = alpaka::access(acc);
    (void) readAcc;
    (void) writeAcc;
    (void) readWriteAcc;

    // constraining from single-tag to single-tag
    alpaka::Accessor<MemoryHandle, int, Size, 1, alpaka::ReadAccess> readAcc2 = alpaka::readAccess(readAcc);
    alpaka::Accessor<MemoryHandle, int, Size, 1, alpaka::WriteAccess> writeAcc2 = alpaka::writeAccess(writeAcc);
    alpaka::Accessor<MemoryHandle, int, Size, 1, alpaka::ReadWriteAccess> readWriteAcc2 = alpaka::access(readWriteAcc);
    (void) readAcc2;
    (void) writeAcc2;
    (void) readWriteAcc2;
}

namespace
{
    struct BufferAccessorKernelRead
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpaka::Accessor<TMemoryHandle, int, TIdx, 1, alpaka::ReadAccess> const r1,
            alpaka::BufferAccessor<TAcc, int, 1, alpaka::ReadAccess> const r2,
            alpaka::BufferAccessor<TAcc, int, 1, alpaka::ReadAccess, TIdx> const r3) const noexcept
        {
            static_assert(std::is_same<decltype(r1), decltype(r2)>::value, "");
            static_assert(std::is_same<decltype(r2), decltype(r3)>::value, "");
        }
    };

    struct BufferAccessorKernelWrite
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpaka::Accessor<TMemoryHandle, int, TIdx, 1, alpaka::WriteAccess> const w1,
            alpaka::BufferAccessor<TAcc, int, 1, alpaka::WriteAccess> const w2,
            alpaka::BufferAccessor<TAcc, int, 1, alpaka::WriteAccess, TIdx> const w3) const noexcept
        {
            static_assert(std::is_same<decltype(w1), decltype(w2)>::value, "");
            static_assert(std::is_same<decltype(w2), decltype(w3)>::value, "");
        }
    };
    struct BufferAccessorKernelReadWrite
    {
        template<typename TAcc, typename TMemoryHandle, typename TIdx>
        ALPAKA_FN_ACC void operator()(
            TAcc const&,
            alpaka::Accessor<TMemoryHandle, int, TIdx, 1, alpaka::ReadWriteAccess> const rw1,
            alpaka::BufferAccessor<TAcc, int, 1> const rw2,
            alpaka::BufferAccessor<TAcc, int, 1, alpaka::ReadWriteAccess> const rw3,
            alpaka::BufferAccessor<TAcc, int, 1, alpaka::ReadWriteAccess, TIdx> const rw4) const noexcept
        {
            static_assert(std::is_same<decltype(rw1), decltype(rw2)>::value, "");
            static_assert(std::is_same<decltype(rw2), decltype(rw3)>::value, "");
            static_assert(std::is_same<decltype(rw3), decltype(rw4)>::value, "");
        }
    };
} // namespace

TEST_CASE("BufferAccessor", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto buffer = alpaka::allocBuf<int, Size>(devAcc, Size{1});

    auto const workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};
    alpaka::exec<Acc>(
        queue,
        workdiv,
        BufferAccessorKernelRead{},
        alpaka::readAccess(buffer),
        alpaka::readAccess(buffer),
        alpaka::readAccess(buffer));
    alpaka::exec<Acc>(
        queue,
        workdiv,
        BufferAccessorKernelWrite{},
        alpaka::writeAccess(buffer),
        alpaka::writeAccess(buffer),
        alpaka::writeAccess(buffer));
    alpaka::exec<Acc>(
        queue,
        workdiv,
        BufferAccessorKernelReadWrite{},
        alpaka::access(buffer),
        alpaka::access(buffer),
        alpaka::access(buffer),
        alpaka::access(buffer));
}
