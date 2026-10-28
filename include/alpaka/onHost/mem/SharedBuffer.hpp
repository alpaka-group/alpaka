/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/mem/View.hpp"
#include "alpaka/mem/concepts/detail/InnerTypeAllowedCast.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/mem/ManagedDealloc.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    /** Life time managed buffer with contiguous data
     *
     * This buffer owns the data and will deallocate it when last copy is destroyed.
     * Const-ness of the buffer instance is propagated to the data region.
     * A copy of this instance will only perform a shallow copy, to perform a deep copy to duplicate the data you
     * should use @c onHost::memcpy.
     */
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment = Alignment<>>
    struct SharedBuffer : View<T_Api, T_Type, T_Extents, T_MemAlignment>
    {
    private:
        using BaseView = View<T_Api, T_Type, T_Extents, T_MemAlignment>;

        /** Constructor with existing managed deleter */
        SharedBuffer(
            T_Api const api,
            T_Type* data,
            T_Extents const& extents,
            T_Extents const& pitches,
            std::shared_ptr<internal::ManagedDealloc> managedDeleter,
            T_MemAlignment const memAlignment)
            : BaseView{api, data, extents, pitches, memAlignment}
            , m_deleter{std::move(managedDeleter)}
        {
        }

        // friend declaration is required that any type of SharedBuffer can access the private constructor
        template<
            alpaka::concepts::Api T_OtherApi,
            typename T_OtherType,
            alpaka::concepts::Vector T_OtherExtents,
            alpaka::concepts::Alignment T_OtherMemAlignment2>
        friend struct SharedBuffer;

        template<
            alpaka::concepts::Api T_OtherApi,
            typename T_OtherType,
            alpaka::concepts::Vector T_OtherExtents,
            alpaka::concepts::Alignment T_OtherMemAlignment2>
        friend std::ostream& operator<<(
            std::ostream& s,
            SharedBuffer<T_OtherApi, T_OtherType, T_OtherExtents, T_OtherMemAlignment2> const& buffer);

    public:
        template<
            alpaka::concepts::HasApi T_Any,
            alpaka::concepts::Vector T_UserExtents,
            alpaka::concepts::Vector T_UserPitches>
        SharedBuffer(
            T_Any const& any,
            T_Type* data,
            T_UserExtents const& extents,
            T_UserPitches const& pitches,
            std::invocable<> auto const& deleter,
            T_MemAlignment const memAlignment = Alignment{})
            : BaseView{any, data, extents, pitches, memAlignment}
            , m_deleter{std::make_shared<internal::ManagedDealloc>(deleter)}
        {
            static_assert(
                isLosslesslyConvertible_v<typename T_UserPitches::value_type, typename T_UserExtents::value_type>,
                "extent type and pitch type must be lossless convertible");
        }

        template<typename T_TypeOther>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_TypeOther>
        SharedBuffer(SharedBuffer<T_Api, T_TypeOther, T_Extents, T_MemAlignment> const& other)
            : BaseView{static_cast<BaseView>(other)}
            , m_deleter(other.m_deleter)
        {
        }

        SharedBuffer(SharedBuffer const&) = default;

        /** Assignment operator keeping const-ness
         *
         * @attention the assign operator is not removing inner const-ness because the type signature is not changed.
         */
        SharedBuffer& operator=(SharedBuffer const& otherSharedBuffer) = default;

        template<typename T_TypeOther>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_TypeOther>
        SharedBuffer(SharedBuffer<T_Api, T_TypeOther, T_Extents, T_MemAlignment>&& other)
            : BaseView{std::move(static_cast<BaseView>(other))}
            , m_deleter(std::move(other.m_deleter))

        {
        }

        SharedBuffer(SharedBuffer&&) = default;

        SharedBuffer& operator=(SharedBuffer&&) = default;

        auto getView() const
        {
            return BaseView::getConstView();
        }

        auto getView()
        {
            return static_cast<BaseView>(*this);
        }

        /** create a read shared buffer view */
        auto getConstSharedBuffer() const
        {
            using ConstValueType = std::add_const_t<typename BaseView::value_type>;
            return SharedBuffer<T_Api, ConstValueType, T_Extents, T_MemAlignment>(
                T_Api{},
                static_cast<ConstValueType*>(this->data()),
                this->getExtents(),
                this->getPitches(),
                m_deleter,
                T_MemAlignment{});
        }

        /** Creates a buffer pointing to a part of the memory.
         *
         * @param extents number of elements for each dimension
         * @return shared buffer which is pointing only to a part of the original buffer.
         */
        auto getSubSharedBuffer(alpaka::concepts::VectorOrScalar auto const& extents) const
        {
            Vec extentMd = extents;
            assert((extentMd <= this->getExtents()).reduce(std::logical_and{}));
            return SharedBuffer<T_Api, std::remove_pointer_t<ALPAKA_TYPEOF(this->data())>, T_Extents, T_MemAlignment>{
                T_Api{},
                this->data(),
                extentMd,
                this->getPitches(),
                m_deleter,
                T_MemAlignment{}};
        }

        auto getSubSharedBuffer(alpaka::concepts::VectorOrScalar auto const& extents)
        {
            Vec extentMd = extents;
            assert((extentMd <= this->getExtents()).reduce(std::logical_and{}));
            return SharedBuffer<T_Api, std::remove_pointer_t<ALPAKA_TYPEOF(this->data())>, T_Extents, T_MemAlignment>{
                T_Api{},
                this->data(),
                extentMd,
                this->getPitches(),
                m_deleter,
                T_MemAlignment{}};
        }

        /** Creates a shared sub-buffer view to a part of the memory.
         *
         * @param offsets offset in elements to the original buffer
         * @param extents number of elements for each dimension
         * @return Buffer which is pointing only to a part of the original buffer with a shifted origin pointer.
         *         Buffer which pointThe alignment of the sub view is reduced to the element alignment.
         */
        auto getSubSharedBuffer(
            alpaka::concepts::VectorOrScalar auto const& offsets,
            alpaka::concepts::VectorOrScalar auto const& extents) const
        {
            Vec offsetMd = offsets;
            Vec extentMd = extents;
            assert((offsetMd + extentMd <= this->getExtents()).reduce(std::logical_and{}));
            auto shiftedPtr = &(*this)[offsetMd];
            return SharedBuffer<T_Api, std::remove_pointer_t<ALPAKA_TYPEOF(shiftedPtr)>, T_Extents, Alignment<>>{
                T_Api{},
                shiftedPtr,
                extentMd,
                this->getPitches(),
                m_deleter,
                Alignment<>{}};
        }

        auto getSubSharedBuffer(
            alpaka::concepts::VectorOrScalar auto const& offsets,
            alpaka::concepts::VectorOrScalar auto const& extents)
        {
            Vec offsetMd = offsets;
            Vec extentMd = extents;
            assert((offsetMd + extentMd <= this->getExtents()).reduce(std::logical_and{}));
            auto shiftedPtr = &(*this)[offsetMd];
            return SharedBuffer<T_Api, std::remove_pointer_t<ALPAKA_TYPEOF(shiftedPtr)>, T_Extents, Alignment<>>{
                T_Api{},
                shiftedPtr,
                extentMd,
                this->getPitches(),
                m_deleter,
                Alignment<>{}};
        }

        /** Adds a destructor action to the shared buffer
         *
         * The action will be executed when the buffer is destroyed.
         * This can be used to add additional cleanup actions e.g. waiting on a specific queue.
         * Actions are executed in FIFO order.
         *
         * @param action callable to execute on destruction
         */
        void addDestructorAction(std::function<void()>&& action)
        {
            m_deleter->addAction(ALPAKA_FORWARD(action));
        }

        /** Add an action to be executed when the shared_ptr is destroyed.
         *
         * @param action Callable to execute on destruction.
         */
        void destructorWaitFor(auto const& any)
        {
            addDestructorAction([any]() { onHost::wait(any); });
        }

        /** Keep the buffer alive until at least the current spot in the queue, even if it runs out of scope.
         * This ensures that the buffer is and stays valid in previously enqueued kernels. There is *no* guarantee
         * that the buffer is deleted immediately when the last reference to it is deleted.
         *
         * This differs from `destructorWaitFor`, because that function waits, while `keepAlive` does not block
         * anything, it just extends lifetime.
         *
         * @param queue The queue to enqueue to.
         */
        void keepAlive(auto& queue)
        {
            // enqueue an empty lambda that keeps a copy of the buffer
            // as long as the copy lives (which is as long as it takes the queue to get to this point), the buffer will
            // stay valid
            auto del = m_deleter;
            queue.enqueueHostFnDeferred([_ = std::move(del)] {});
        }

        /** Return the number of SharedBuffers which points to the same memory */
        [[nodiscard]] constexpr long getUseCount() const noexcept
        {
            return m_deleter.use_count();
        }

        /** True if SharedBuffer is pointing to valid memory. */
        [[nodiscard]] constexpr explicit operator bool() const noexcept
        {
            return static_cast<bool>(m_deleter);
        }

    private:
        /** @todo move this to traits or somewhere else that it can be used everywhere */
        template<alpaka::concepts::Pointer T>
        using ConstPtr_t = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<T>>>;

        std::shared_ptr<internal::ManagedDealloc> m_deleter;
    }; // namespace alpaka::onHost

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches,
        alpaka::concepts::Alignment T_MemAlignment>
    SharedBuffer(
        T_Any const&,
        T_Type*,
        T_UserExtents const&,
        T_UserPitches const&,
        std::invocable<> auto,
        T_MemAlignment const)
        -> SharedBuffer<
            ALPAKA_TYPEOF(getApi(std::declval<T_Any>())),
            T_Type,
            typename T_UserPitches::UniVec,
            T_MemAlignment>;

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches>
    SharedBuffer(T_Any const&, T_Type*, T_UserExtents const&, T_UserPitches const&, std::invocable<> auto)
        -> SharedBuffer<
            ALPAKA_TYPEOF(getApi(std::declval<T_Any>())),
            T_Type,
            typename T_UserPitches::UniVec,
            Alignment<>>;

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct MakeAccessibleOnAcc::Op<SharedBuffer<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        auto operator()(auto&& any) const
        {
            return any.getView();
        }
    };

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    std::ostream& operator<<(std::ostream& s, SharedBuffer<T_Api, T_Type, T_Extents, T_MemAlignment> const& buff)
    {
        return s << "SharedBuffer{ dim=" << ALPAKA_TYPEOF(buff)::dim() << ", api= " << onHost::getName(T_Api{})
                 << ", extents=" << buff.getExtents().toString() << ", pitches=" << buff.getPitches().toString()
                 << " , alignment=" << T_MemAlignment::template get<T_Type>() << " }";
    }

} // namespace alpaka::onHost

namespace alpaka::internal
{
    // external define the API trait to support constexpr evaluation
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<onHost::SharedBuffer<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        inline constexpr auto operator()(auto&& buffer) const
        {
            alpaka::unused(buffer);
            return T_Api{};
        }
    };

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct CopyConstructableDataSource<onHost::SharedBuffer<T_Api, T_Type, T_Extents, T_MemAlignment>> : std::true_type
    {
        using InnerMutable = onHost::SharedBuffer<T_Api, std::remove_const_t<T_Type>, T_Extents, T_MemAlignment>;
        using InnerConst = onHost::SharedBuffer<T_Api, std::add_const_t<T_Type>, T_Extents, T_MemAlignment>;
    };

} // namespace alpaka::internal
