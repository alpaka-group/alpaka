/* Copyright 2021 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

#include <complex>
#include <iostream>

namespace alpaka
{
    //! Implementation of a complex number useable on host and device.
    //!
    //! It follows the layout of std::complex and so array-oriented access.
    //! Inside the class template implements all methods and operators as std::complex<T>.
    //! Additionally, provides an implicit conversion to and from std::complex<T>.
    //! All methods besides operators << and >> are host-device.
    //! Note that it does not provide non-member functions of std::complex besides the operators.
    //! Those are provided the same way as alpaka math functions for real numbers.
    //!
    //! Note that unlike most of alpaka, this is a concrete type template, and not merely a concept.
    //!
    //! Naming and order of the methods match https://en.cppreference.com/w/cpp/numeric/complex.
    //!
    //! @tparam T type of the real and imaginary part: float, double, or long double.
    template<typename T>
    class Complex
    {
    public:
        //! Type of the real and imaginary parts
        using value_type = T;

        //! Constructor from the given real and imaginary parts
        constexpr ALPAKA_FN_HOST_ACC Complex(T const& real = T{}, T const& imag = T{}) : m_real(real), m_imag(imag)
        {
        }

        //! Copy constructor
        constexpr ALPAKA_FN_HOST_ACC Complex(Complex const& other) = default;

        //! Constructor from Complex of another type
        template<typename U>
        constexpr ALPAKA_FN_HOST_ACC Complex(Complex<U> const& other)
            : m_real(static_cast<T>(other.real()))
            , m_imag(static_cast<T>(other.imag()))
        {
        }

        //! Constructor from std::complex
        constexpr ALPAKA_FN_HOST_ACC Complex(std::complex<T> const& other) : m_real(other.real()), m_imag(other.imag())
        {
        }

        //! Conversion to std::complex
        constexpr ALPAKA_FN_HOST_ACC operator std::complex<T>() const
        {
            return std::complex<T>{m_real, m_imag};
        }

        //! Assignment
        ALPAKA_FN_HOST_ACC Complex& operator=(Complex const&) = default;

        //! Get the real part
        constexpr ALPAKA_FN_HOST_ACC T real() const
        {
            return m_real;
        }

        //! Stub for get the real part (does nothing, added for compatibility with std::complex)
        constexpr ALPAKA_FN_HOST_ACC void real(T)
        {
        }

        //! Get the imaginary part
        constexpr ALPAKA_FN_HOST_ACC T imag() const
        {
            return m_imag;
        }

        //! Stub for get the imaginary part (does nothing, added for compatibility with std::complex)
        constexpr ALPAKA_FN_HOST_ACC void imag(T)
        {
        }

        //! Addition assignment with a real number
        ALPAKA_FN_HOST_ACC Complex& operator+=(T const& other)
        {
            m_real += other;
            return *this;
        }

        //! Addition assignment with a complex number
        template<typename U>
        ALPAKA_FN_HOST_ACC Complex& operator+=(Complex<U> const& other)
        {
            m_real += static_cast<T>(other.real());
            m_imag += static_cast<T>(other.imag());
            return *this;
        }

        //! Subtraction assignment with a real number
        ALPAKA_FN_HOST_ACC Complex& operator-=(T const& other)
        {
            m_real -= other;
            return *this;
        }

        //! Subtraction assignment with a complex number
        template<typename U>
        ALPAKA_FN_HOST_ACC Complex& operator-=(Complex<U> const& other)
        {
            m_real -= static_cast<T>(other.real());
            m_imag -= static_cast<T>(other.imag());
            return *this;
        }

        //! Multiplication assignment with a real number
        ALPAKA_FN_HOST_ACC Complex& operator*=(T const& other)
        {
            m_real *= other;
            m_imag *= other;
            return *this;
        }

        //! Multiplication assignment with a complex number
        template<typename U>
        ALPAKA_FN_HOST_ACC Complex& operator*=(Complex<U> const& other)
        {
            auto const newReal = m_real * static_cast<T>(other.real()) - m_imag * static_cast<T>(other.imag());
            auto const newImag = m_imag * static_cast<T>(other.real()) + m_real * static_cast<T>(other.imag());
            m_real = newReal;
            m_imag = newImag;
            return *this;
        }

        //! Division assignment with a real number
        ALPAKA_FN_HOST_ACC Complex& operator/=(T const& other)
        {
            m_real /= other;
            m_imag /= other;
            return *this;
        }

        //! Division assignment with a complex number
        template<typename U>
        ALPAKA_FN_HOST_ACC Complex& operator/=(Complex<U> const& other)
        {
            return *this *= Complex{
                       static_cast<T>(other.real() / (other.real() * other.real() + other.imag() * other.imag())),
                       static_cast<T>(-other.imag() / (other.real() * other.real() + other.imag() * other.imag()))};
        }

    private:
        //! Real and imaginary parts, storage enables array-oriented access
        T m_real, m_imag;
    };

    //! Unary plus (added for compatibility with std::complex)
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator+(Complex<T> const& val)
    {
        return val;
    }

    //! Unary minus
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator-(Complex<T> const& val)
    {
        return Complex<T>{-val.real(), -val.imag()};
    }

    //! Addition of two complex numbers
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator+(Complex<T> const& lhs, Complex<T> const& rhs)
    {
        return Complex<T>{lhs.real() + rhs.real(), lhs.imag() + rhs.imag()};
    }

    //! Addition of a complex and a real number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator+(Complex<T> const& lhs, T const& rhs)
    {
        return Complex<T>{lhs.real() + rhs, lhs.imag()};
    }

    //! Addition of a real and a complex number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator+(T const& lhs, Complex<T> const& rhs)
    {
        return Complex<T>{lhs + rhs.real(), rhs.imag()};
    }

    //! Subtraction of two complex numbers
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator-(Complex<T> const& lhs, Complex<T> const& rhs)
    {
        return Complex<T>{lhs.real() - rhs.real(), lhs.imag() - rhs.imag()};
    }

    //! Subtraction of a complex and a real number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator-(Complex<T> const& lhs, T const& rhs)
    {
        return Complex<T>{lhs.real() - rhs, lhs.imag()};
    }

    //! Subtraction of a real and a complex number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator-(T const& lhs, Complex<T> const& rhs)
    {
        return Complex<T>{lhs - rhs.real(), -rhs.imag()};
    }

    //! Muptiplication of two complex numbers
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator*(Complex<T> const& lhs, Complex<T> const& rhs)
    {
        return Complex<T>{
            lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
            lhs.imag() * rhs.real() + lhs.real() * rhs.imag()};
    }

    //! Muptiplication of a complex and a real number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator*(Complex<T> const& lhs, T const& rhs)
    {
        return Complex<T>{lhs.real() * rhs, lhs.imag() * rhs};
    }

    //! Muptiplication of a real and a complex number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator*(T const& lhs, Complex<T> const& rhs)
    {
        return Complex<T>{lhs * rhs.real(), lhs * rhs.imag()};
    }

    //! Division of two complex numbers
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator/(Complex<T> const& lhs, Complex<T> const& rhs)
    {
        return lhs
            * Complex<T>{
                rhs.real() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag()),
                -rhs.imag() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag())};
    }

    //! Division of complex and a real number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator/(Complex<T> const& lhs, T const& rhs)
    {
        return Complex<T>{lhs.real() / rhs, lhs.imag() / rhs};
    }

    //! Division of a real and a complex number
    template<typename T>
    ALPAKA_FN_HOST_ACC Complex<T> operator/(T const& lhs, Complex<T> const& rhs)
    {
        return lhs
            * Complex<T>{
                rhs.real() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag()),
                -rhs.imag() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag())};
    }

    //! Equality of two complex numbers
    template<typename T>
    constexpr ALPAKA_FN_HOST_ACC bool operator==(Complex<T> const& lhs, Complex<T> const& rhs)
    {
        return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
    }

    //! Equality of a complex and a real number
    template<typename T>
    constexpr ALPAKA_FN_HOST_ACC bool operator==(Complex<T> const& lhs, T const& rhs)
    {
        return (lhs.real() == rhs) && (lhs.imag() == static_cast<T>(0));
    }

    //! Equality of a real and a complex number
    template<typename T>
    constexpr ALPAKA_FN_HOST_ACC bool operator==(T const& lhs, Complex<T> const& rhs)
    {
        return (lhs == rhs.real()) && (static_cast<T>(0) == rhs.imag());
    }

    //! Inequality of two complex numbers.
    //!
    //! @note this and other versions of operator != should be removed since C++20, as so does std::complex
    template<typename T>
    constexpr ALPAKA_FN_HOST_ACC bool operator!=(Complex<T> const& lhs, Complex<T> const& rhs)
    {
        return (lhs.real() != rhs.real()) || (lhs.imag() != rhs.imag());
    }

    //! Inequality of a complex and a real number
    template<typename T>
    constexpr ALPAKA_FN_HOST_ACC bool operator!=(Complex<T> const& lhs, T const& rhs)
    {
        return (lhs.real() != rhs) || (lhs.imag() != static_cast<T>(0));
    }

    //! Inequality of a real and a complex number
    template<typename T>
    constexpr ALPAKA_FN_HOST_ACC bool operator!=(T const& lhs, Complex<T> const& rhs)
    {
        return (lhs != rhs.real()) || (static_cast<T>(0) != rhs.imag());
    }

    //! Host-only output of a complex number
    template<typename T, typename TChar, typename TTraits>
    std::basic_ostream<TChar, TTraits>& operator<<(std::basic_ostream<TChar, TTraits>& os, Complex<T> const& x)
    {
        os << std::complex<T>{x};
        return os;
    }

    //! Host-only input of a complex number
    template<typename T, typename TChar, typename TTraits>
    std::basic_istream<TChar, TTraits>& operator>>(std::basic_istream<TChar, TTraits>& is, Complex<T> const& x)
    {
        std::complex<T> z;
        is >> z;
        x = z;
        return is;
    }

} // namespace alpaka
