/**
 * \file
 * Copyright 2018 Jonas Schenke
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <alpaka/alpaka.hpp>

//#############################################################################
//! An iterator base class.
//!
//! \tparam T The type.
//! \tparam TBuf The buffer type (standard is T).
template <typename T, typename TBuf = T>
class Iterator
{
protected:
    const TBuf *data;
    uint64_t index;
    const uint64_t maximum;

public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param data A pointer to the data.
    //! \param linearizedIndex The linearized index.
    //! \param n The problem size.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Iterator(const TBuf *data,
                                                 uint32_t index,
                                                 uint64_t maximum)
        : data(data), index(index), maximum(maximum)
    {
    }

    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param other The other iterator object.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Iterator(const Iterator &other) = default;

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns true if objects are equal and false otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
    operator==(const Iterator &other) const -> bool
    {
        return (this->data == other.data) && (this->index == other.index) &&
               (this->maximum == other.maximum);
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns false if objects are equal and true otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
    operator!=(const Iterator &other) const -> bool
    {
        return !operator==(other);
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns false if the other object is equal or smaller and true
    //! otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
    operator<(const Iterator &other) const -> bool
    {
        return index < other.index;
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns false if the other object is equal or bigger and true otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
    operator>(const Iterator &other) const -> bool
    {
        return index > other.index;
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns true if the other object is equal or bigger and false otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
    operator<=(const Iterator &other) const -> bool
    {
        return index <= other.index;
    }

    //-----------------------------------------------------------------------------
    //! Compare operator.
    //!
    //! \param other The other object.
    //!
    //! Returns true if the other object is equal or smaller and false
    //! otherwise.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto
    operator>=(const Iterator &other) const -> bool
    {
        return index >= other.index;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator*() -> const T &
    {
        return data[index];
    }
};

//#############################################################################
//! A CPU memory iterator.
//!
//! \tparam TAcc The accelerator type.
//! \tparam T The type.
//! \tparam TBuf The buffer type (standard is T).
template <typename TAcc, typename T, typename TBuf = T>
class IteratorCpu : public Iterator<T, TBuf>
{
public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param acc The accelerator object.
    //! \param data A pointer to the data.
    //! \param linearizedIndex The linearized index.
    //! \param gridSize The grid size.
    //! \param n The problem size.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE IteratorCpu(const TAcc &acc,
                                                    const TBuf *data,
                                                    uint32_t linearizedIndex,
                                                    uint32_t gridSize,
                                                    uint64_t n)
        : Iterator<T, TBuf>(
              data,
              (n * linearizedIndex) /
                  alpaka::math::min(acc, static_cast<uint64_t>(gridSize), n),
              (n * (linearizedIndex + 1)) /
                  alpaka::math::min(acc, static_cast<uint64_t>(gridSize), n))
    {
    }

    //-----------------------------------------------------------------------------
    //! Returns the iterator for the last item.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto end() const -> IteratorCpu
    {
        IteratorCpu ret(*this);
        ret.index = this->maximum;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Increments the internal pointer to the next one and returns this
    //! element.
    //!
    //! Returns a reference to the next index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++() -> IteratorCpu &
    {
        ++(this->index);
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and increments the internal pointer to the
    //! next one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++(int) -> IteratorCpu
    {
        auto ret(*this);
        ++(this->index);
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Decrements the internal pointer to the previous one and returns the this
    //! element.
    //!
    //! Returns a reference to the previous index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--() -> IteratorCpu &
    {
        --(this->index);
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and decrements the internal pointer to the
    //! previous one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--(int) -> IteratorCpu
    {
        auto ret(*this);
        --(this->index);
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index + a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+(uint64_t n) const
        -> IteratorCpu
    {
        IteratorCpu ret(*this);
        ret.index += n;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index - a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-(uint64_t n) const
        -> IteratorCpu
    {
        IteratorCpu ret(*this);
        ret.index -= n;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Addition assignment.
    //!
    //! \param other The other object.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+=(uint64_t offset)
        -> IteratorCpu &
    {
        this->index += offset;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Substraction assignment.
    //!
    //! \param other The other object.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-=(uint64_t offset)
        -> IteratorCpu &
    {
        this->index -= offset;
        return *this;
    }
};

//#############################################################################
//! A GPU memory iterator.
//!
//! \tparam TAcc The accelerator type.
//! \tparam T The type.
//! \tparam TBuf The buffer type (standard is T).
template <typename TAcc, typename T, typename TBuf = T>
class IteratorGpu : public Iterator<T, TBuf>
{
private:
    const uint32_t gridSize;

public:
    //-----------------------------------------------------------------------------
    //! Constructor.
    //!
    //! \param acc The accelerator object.
    //! \param data A pointer to the data.
    //! \param linearizedIndex The linearized index.
    //! \param gridSize The grid size.
    //! \param n The problem size.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE IteratorGpu(const TAcc &acc,
                                                    const TBuf *data,
                                                    uint32_t linearizedIndex,
                                                    uint32_t gridSize,
                                                    uint64_t n)
        : Iterator<T, TBuf>(data, linearizedIndex, n), gridSize(gridSize)
    {
    }

    //-----------------------------------------------------------------------------
    //! Returns the iterator for the last item.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto end() const -> IteratorGpu
    {
        IteratorGpu ret(*this);
        ret.index = this->maximum;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Increments the internal pointer to the next one and returns this
    //! element.
    //!
    //! Returns a reference to the next index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++() -> IteratorGpu &
    {
        this->index += this->gridSize;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and increments the internal pointer to the
    //! next one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator++(int) -> IteratorGpu
    {
        auto ret(*this);
        this->index += this->gridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Decrements the internal pointer to the previous one and returns the this
    //! element.
    //!
    //! Returns a reference to the previous index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--() -> IteratorGpu &
    {
        this->index -= this->gridSize;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Returns the current element and decrements the internal pointer to the
    //! previous one.
    //!
    //! Returns a reference to the current index.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator--(int) -> IteratorGpu
    {
        auto ret(*this);
        this->index -= this->gridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index + a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+(uint64_t n) const
        -> IteratorGpu
    {
        auto ret(*this);
        ret.index += n * gridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Returns the index - a supplied offset.
    //!
    //! \param n The offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-(uint64_t n) const
        -> IteratorGpu
    {
        auto ret(*this);
        ret.index -= n * gridSize;
        return ret;
    }

    //-----------------------------------------------------------------------------
    //! Addition assignment.
    //!
    //! \param other The other object.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator+=(uint64_t offset)
        -> IteratorGpu &
    {
        this->index += offset * this->gridSize;
        return *this;
    }

    //-----------------------------------------------------------------------------
    //! Substraction assignment.
    //!
    //! \param other The other object.
    //!
    //! Returns the current object offset by the offset.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator-=(uint64_t offset)
        -> IteratorGpu &
    {
        this->index -= offset * this->gridSize;
        return *this;
    }
};
