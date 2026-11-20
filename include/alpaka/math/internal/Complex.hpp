/* Copyright 2024 Sergei Bastrakov, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/math.hpp"
#include "alpaka/math/floatEqualExact.hpp"
#include "alpaka/trait.hpp"
#include "math.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>

namespace alpaka::math
{
    namespace internal
    {
        //! Implementation of a complex number usable on host and device.
        //!
        //! It follows the layout of std::complex and so array-oriented access.
        //! The class template implements all methods and operators as std::complex<T>.
        //! Additionally, it provides an implicit conversion to and from std::complex<T>.
        //! All methods besides operators << and >> are host-device.
        //! It does not provide non-member functions of std::complex besides the operators.
        //! Those are provided the same way as alpaka math functions for real numbers.
        //!
        //! Note that unlike most of alpaka, this is a concrete type template, and not merely a concept.
        //!
        //! Naming and order of the methods match https://en.cppreference.com/w/cpp/numeric/complex in C++17.
        //! Implementation chose to not extend it e.g. by adding constexpr to some places that would get it in C++20.
        //! The motivation is that with internal conversion to std::complex<T> for CPU backends, it would define the
        //! common interface for generic code anyways. So it is more clear to have alpaka's interface exactly matching
        //! when possible, and not "improving".
        //!
        //! @tparam T type of the real and imaginary part: float, double, or long double.
        template<typename T>
        class Complex
        {
        public:
            // Make sure the input type is floating-point
            static_assert(std::is_floating_point_v<T>);

            //! Type of the real and imaginary parts
            using value_type = T;

            //! Constructor from the given real and imaginary parts
            constexpr Complex(T const& real = T{}, T const& imag = T{}) : m_real(real), m_imag(imag)
            {
            }

            //! Copy constructor
            constexpr Complex(Complex const& other) = default;

            //! Constructor from Complex of another type
            template<typename U>
            constexpr Complex(Complex<U> const& other)
                : m_real(static_cast<T>(other.real()))
                , m_imag(static_cast<T>(other.imag()))
            {
            }

            //! Constructor from std::complex
            constexpr Complex(std::complex<T> const& other) : m_real(other.real()), m_imag(other.imag())
            {
            }

            //! Conversion to std::complex
            constexpr operator std::complex<T>() const
            {
                return std::complex<T>{m_real, m_imag};
            }

            //! Assignment
            Complex& operator=(Complex const&) = default;

            //! Get the real part
            constexpr T real() const
            {
                return m_real;
            }

            //! Set the real part
            constexpr void real(T value)
            {
                m_real = value;
            }

            //! Get the imaginary part
            constexpr T imag() const
            {
                return m_imag;
            }

            //! Set the imaginary part
            constexpr void imag(T value)
            {
                m_imag = value;
            }

            //! Addition assignment with a real number
            constexpr Complex& operator+=(T const& other)
            {
                m_real += other;
                return *this;
            }

            //! Addition assignment with a complex number
            template<typename U>
            constexpr Complex& operator+=(Complex<U> const& other)
            {
                m_real += static_cast<T>(other.real());
                m_imag += static_cast<T>(other.imag());
                return *this;
            }

            //! Subtraction assignment with a real number
            constexpr Complex& operator-=(T const& other)
            {
                m_real -= other;
                return *this;
            }

            //! Subtraction assignment with a complex number
            template<typename U>
            constexpr Complex& operator-=(Complex<U> const& other)
            {
                m_real -= static_cast<T>(other.real());
                m_imag -= static_cast<T>(other.imag());
                return *this;
            }

            //! Multiplication assignment with a real number
            constexpr Complex& operator*=(T const& other)
            {
                m_real *= other;
                m_imag *= other;
                return *this;
            }

            //! Multiplication assignment with a complex number
            template<typename U>
            constexpr Complex& operator*=(Complex<U> const& other)
            {
                auto const newReal = m_real * static_cast<T>(other.real()) - m_imag * static_cast<T>(other.imag());
                auto const newImag = m_imag * static_cast<T>(other.real()) + m_real * static_cast<T>(other.imag());
                m_real = newReal;
                m_imag = newImag;
                return *this;
            }

            //! Division assignment with a real number
            constexpr Complex& operator/=(T const& other)
            {
                m_real /= other;
                m_imag /= other;
                return *this;
            }

            //! Division assignment with a complex number
            template<typename U>
            constexpr Complex& operator/=(Complex<U> const& other)
            {
                return *this *= Complex{
                           static_cast<T>(other.real() / (other.real() * other.real() + other.imag() * other.imag())),
                           static_cast<T>(
                               -other.imag() / (other.real() * other.real() + other.imag() * other.imag()))};
            }

        private:
            //! Real and imaginary parts, storage enables array-oriented access
            T m_real, m_imag;
        };

        //! Host-device arithmetic operations matching std::complex<T>.
        //!
        //! They take and return alpaka::math::Complex.
        //!
        //! @{
        //!

        //! Unary plus (added for compatibility with std::complex)
        template<typename T>
        constexpr Complex<T> operator+(Complex<T> const& val)
        {
            return val;
        }

        //! Unary minus
        template<typename T>
        constexpr Complex<T> operator-(Complex<T> const& val)
        {
            return Complex<T>{-val.real(), -val.imag()};
        }

        //! Addition of two complex numbers
        template<typename T>
        constexpr Complex<T> operator+(Complex<T> const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{lhs.real() + rhs.real(), lhs.imag() + rhs.imag()};
        }

        //! Addition of a complex and a real number
        template<typename T>
        constexpr Complex<T> operator+(Complex<T> const& lhs, T const& rhs)
        {
            return Complex<T>{lhs.real() + rhs, lhs.imag()};
        }

        //! Addition of a real and a complex number
        template<typename T>
        constexpr Complex<T> operator+(T const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{lhs + rhs.real(), rhs.imag()};
        }

        //! Subtraction of two complex numbers
        template<typename T>
        constexpr Complex<T> operator-(Complex<T> const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{lhs.real() - rhs.real(), lhs.imag() - rhs.imag()};
        }

        //! Subtraction of a complex and a real number
        template<typename T>
        constexpr Complex<T> operator-(Complex<T> const& lhs, T const& rhs)
        {
            return Complex<T>{lhs.real() - rhs, lhs.imag()};
        }

        //! Subtraction of a real and a complex number
        template<typename T>
        constexpr Complex<T> operator-(T const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{lhs - rhs.real(), -rhs.imag()};
        }

        //! Muptiplication of two complex numbers
        template<typename T>
        constexpr Complex<T> operator*(Complex<T> const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{
                lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                lhs.imag() * rhs.real() + lhs.real() * rhs.imag()};
        }

        //! Muptiplication of a complex and a real number
        template<typename T>
        constexpr Complex<T> operator*(Complex<T> const& lhs, T const& rhs)
        {
            return Complex<T>{lhs.real() * rhs, lhs.imag() * rhs};
        }

        //! Muptiplication of a real and a complex number
        template<typename T>
        constexpr Complex<T> operator*(T const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{lhs * rhs.real(), lhs * rhs.imag()};
        }

        //! Division of two complex numbers
        template<typename T>
        constexpr Complex<T> operator/(Complex<T> const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{
                (lhs.real() * rhs.real() + lhs.imag() * rhs.imag())
                    / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag()),
                (lhs.imag() * rhs.real() - lhs.real() * rhs.imag())
                    / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag())};
        }

        //! Division of complex and a real number
        template<typename T>
        constexpr Complex<T> operator/(Complex<T> const& lhs, T const& rhs)
        {
            return Complex<T>{lhs.real() / rhs, lhs.imag() / rhs};
        }

        //! Division of a real and a complex number
        template<typename T>
        constexpr Complex<T> operator/(T const& lhs, Complex<T> const& rhs)
        {
            return Complex<T>{
                lhs * rhs.real() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag()),
                -lhs * rhs.imag() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag())};
        }

        //! Equality of two complex numbers
        template<typename T>
        constexpr bool operator==(Complex<T> const& lhs, Complex<T> const& rhs)
        {
            return math::floatEqualExactNoWarning(lhs.real(), rhs.real())
                   && math::floatEqualExactNoWarning(lhs.imag(), rhs.imag());
        }

        //! Equality of a complex and a real number
        template<typename T>
        constexpr bool operator==(Complex<T> const& lhs, T const& rhs)
        {
            return math::floatEqualExactNoWarning(lhs.real(), rhs)
                   && math::floatEqualExactNoWarning(lhs.imag(), static_cast<T>(0));
        }

        //! Equality of a real and a complex number
        template<typename T>
        constexpr bool operator==(T const& lhs, Complex<T> const& rhs)
        {
            return math::floatEqualExactNoWarning(lhs, rhs.real())
                   && math::floatEqualExactNoWarning(static_cast<T>(0), rhs.imag());
        }

        //! Inequality of two complex numbers.
        //!
        //! @note this and other versions of operator != should be removed since C++20, as so does std::complex
        template<typename T>
        constexpr bool operator!=(Complex<T> const& lhs, Complex<T> const& rhs)
        {
            return !(lhs == rhs);
        }

        //! Inequality of a complex and a real number
        template<typename T>
        constexpr bool operator!=(Complex<T> const& lhs, T const& rhs)
        {
            return !math::floatEqualExactNoWarning(lhs.real(), rhs)
                   || !math::floatEqualExactNoWarning(lhs.imag(), static_cast<T>(0));
        }

        //! Inequality of a real and a complex number
        template<typename T>
        constexpr bool operator!=(T const& lhs, Complex<T> const& rhs)
        {
            return !math::floatEqualExactNoWarning(lhs, rhs.real())
                   || !math::floatEqualExactNoWarning(static_cast<T>(0), rhs.imag());
        }

        //! @}

        //! Host-only output of a complex number
        template<typename T, typename T_Char, typename T_Traits>
        std::basic_ostream<T_Char, T_Traits>& operator<<(std::basic_ostream<T_Char, T_Traits>& os, Complex<T> const& x)
        {
            os << x.operator std::complex<T>();
            return os;
        }

        //! Host-only input of a complex number
        template<typename T, typename T_Char, typename T_Traits>
        std::basic_istream<T_Char, T_Traits>& operator>>(std::basic_istream<T_Char, T_Traits>& is, Complex<T> const& x)
        {
            std::complex<T> z;
            is >> z;
            x = z;
            return is;
        }

        //! Host-only math functions matching std::complex<T>.
        //!
        //! Due to issue #1688, these functions are technically marked host-device and suppress related warnings.
        //! However, they must be called for host only.
        //!
        //! They take and return alpaka::math::Complex (or a real number when appropriate).
        //! Internally cast, fall back to std::complex implementation and cast back.
        //! These functions can be used directly on the host side.
        //! They are also picked up by ADL in math traits for CPU backends.
        //!
        //! On the device side, alpaka math traits must be used instead.
        //! Note that the set of the traits is currently a bit smaller.
        //!
        //! @{
        //!

        //! Absolute value
        template<typename T>
        constexpr T abs(Complex<T> const& x)
        {
            return std::abs(std::complex<T>(x));
        }

        //! Arc cosine
        template<typename T>
        constexpr Complex<T> acos(Complex<T> const& x)
        {
            return std::acos(std::complex<T>(x));
        }

        //! Arc hyperbolic cosine
        template<typename T>
        constexpr Complex<T> acosh(Complex<T> const& x)
        {
            return std::acosh(std::complex<T>(x));
        }

        //! Argument
        template<typename T>
        constexpr T arg(Complex<T> const& x)
        {
            return std::arg(std::complex<T>(x));
        }

        //! Arc sine
        template<typename T>
        constexpr Complex<T> asin(Complex<T> const& x)
        {
            return std::asin(std::complex<T>(x));
        }

        //! Arc hyperbolic sine
        template<typename T>
        constexpr Complex<T> asinh(Complex<T> const& x)
        {
            return std::asinh(std::complex<T>(x));
        }

        //! Arc tangent
        template<typename T>
        constexpr Complex<T> atan(Complex<T> const& x)
        {
            return std::atan(std::complex<T>(x));
        }

        //! Arc hyperbolic tangent
        template<typename T>
        constexpr Complex<T> atanh(Complex<T> const& x)
        {
            return std::atanh(std::complex<T>(x));
        }

        //! Complex conjugate
        template<typename T>
        constexpr Complex<T> conj(Complex<T> const& x)
        {
            return std::conj(std::complex<T>(x));
        }

        //! Cosine
        template<typename T>
        constexpr Complex<T> cos(Complex<T> const& x)
        {
            return std::cos(std::complex<T>(x));
        }

        //! Hyperbolic cosine
        template<typename T>
        constexpr Complex<T> cosh(Complex<T> const& x)
        {
            return std::cosh(std::complex<T>(x));
        }

        //! Exponential
        template<typename T>
        constexpr Complex<T> exp(Complex<T> const& x)
        {
            return std::exp(std::complex<T>(x));
        }

        //! Natural logarithm
        template<typename T>
        constexpr Complex<T> log(Complex<T> const& x)
        {
            return std::log(std::complex<T>(x));
        }

        //! Base 10 logarithm
        template<typename T>
        constexpr Complex<T> log10(Complex<T> const& x)
        {
            return std::log10(std::complex<T>(x));
        }

        //! Squared magnitude
        template<typename T>
        constexpr T norm(Complex<T> const& x)
        {
            return std::norm(std::complex<T>(x));
        }

        //! Get a complex number with given magnitude and phase angle
        template<typename T>
        constexpr Complex<T> polar(T const& r, T const& theta = T())
        {
            return std::polar(r, theta);
        }

        //! Complex power of a complex number
        template<typename T, typename U>
        constexpr auto pow(Complex<T> const& x, Complex<U> const& y)
        {
            // Use same type promotion as std::pow
            auto const result = std::pow(std::complex<T>(x), std::complex<U>(y));
            using ValueType = typename decltype(result)::value_type;
            return Complex<ValueType>(result);
        }

        //! Real power of a complex number
        template<typename T, typename U>
        constexpr auto pow(Complex<T> const& x, U const& y)
        {
            return pow(x, Complex<U>(y));
        }

        //! Complex power of a real number
        template<typename T, typename U>
        constexpr auto pow(T const& x, Complex<U> const& y)
        {
            return pow(Complex<T>(x), y);
        }

        //! Projection onto the Riemann sphere
        template<typename T>
        constexpr Complex<T> proj(Complex<T> const& x)
        {
            return std::proj(std::complex<T>(x));
        }

        //! Sine
        template<typename T>
        constexpr Complex<T> sin(Complex<T> const& x)
        {
            return std::sin(std::complex<T>(x));
        }

        //! Hyperbolic sine
        template<typename T>
        constexpr Complex<T> sinh(Complex<T> const& x)
        {
            return std::sinh(std::complex<T>(x));
        }

        //! Square root
        template<typename T>
        constexpr Complex<T> sqrt(Complex<T> const& x)
        {
            return std::sqrt(std::complex<T>(x));
        }

        //! Tangent
        template<typename T>
        constexpr Complex<T> tan(Complex<T> const& x)
        {
            return std::tan(std::complex<T>(x));
        }

        //! Hyperbolic tangent
        template<typename T>
        constexpr Complex<T> tanh(Complex<T> const& x)
        {
            return std::tanh(std::complex<T>(x));
        }

        //! @}
    } // namespace internal

    using internal::Complex;

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP || ALPAKA_LANG_SYCL

    namespace internal
    {
        template<typename T_MathImpl, typename T_Arg>
        struct Abs::Op<T_MathImpl, Complex<T_Arg>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T_Arg> const& arg) const
            {
                return math::sqrt(arg.real() * arg.real() + arg.imag() * arg.imag());
            }
        };

        //! The acos trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Acos::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // This holds everywhere, including the branch cuts: acos(z) = -i * ln(z + i * sqrt(1 - z^2))
                return Complex<T>{static_cast<T>(0.0), static_cast<T>(-1.0)}
                       * math::log(
                           arg
                           + Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)}
                                 * math::sqrt(static_cast<T>(1.0) - arg * arg));
            }
        };

        //! The acosh trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Acosh::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // acos(z) = ln(z + sqrt(z-1) * sqrt(z+1))
                return math::log(arg + math::sqrt(arg - static_cast<T>(1.0)) * math::sqrt(arg + static_cast<T>(1.0)));
            }
        };

        //! The arg Complex<T> specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Arg::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& argument) const
            {
                return math::atan2(argument.imag(), argument.real());
            }
        };

        //! The asin trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Asin::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // This holds everywhere, including the branch cuts: asin(z) = i * ln(sqrt(1 - z^2) - i * z)
                return Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)}
                       * math::log(
                           math::sqrt(static_cast<T>(1.0) - arg * arg)
                           - Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)} * arg);
            }
        };

        //! The asinh trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Asinh::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // asinh(z) = ln(z + sqrt(z^2 + 1))
                return math::log(arg + math::sqrt(arg * arg + static_cast<T>(1.0)));
            }
        };

        //! The atan trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Atan::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // This holds everywhere, including the branch cuts: atan(z) = -i/2 * ln((i - z) / (i + z))
                return Complex<T>{static_cast<T>(0.0), static_cast<T>(-0.5)}
                       * math::log(
                           (Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)} - arg)
                           / (Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)} + arg));
            }
        };

        //! The atanh trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Atanh::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                //  atanh(z) = 0.5 * (ln(1 + z) - ln(1 - z))
                return static_cast<T>(0.5)
                       * (math::log(static_cast<T>(1.0) + arg) - math::log(static_cast<T>(1.0) - arg));
            }
        };

        //! The conj specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Conj::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl const& /* conj_ctx */, Complex<T> const& arg) const
            {
                return Complex<T>{arg.real(), -arg.imag()};
            }
        };

        //! The cos trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Cos::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // cos(z) = 0.5 * (exp(i * z) + exp(-i * z))
                return T(0.5)
                       * (math::exp(Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)} * arg)
                          + math::exp(Complex<T>{static_cast<T>(0.0), static_cast<T>(-1.0)} * arg));
            }
        };

        //! The cosh trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Cosh::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // cosh(z) = 0.5 * (exp(z) + exp(-z))
                return T(0.5) * (math::exp(arg) + math::exp(static_cast<T>(-1.0) * arg));
            }
        };

        //! The exp trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Exp::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // exp(z) = exp(x + iy) = exp(x) * (cos(y) + i * sin(y))
                T re, im;
                math::sincos(arg.imag(), im, re);
                return math::exp(arg.real()) * Complex<T>{re, im};
            }
        };

        //! The log trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Log::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& argument) const
            {
                // Branch cut along the negative real axis (same as for std::complex),
                // principal value of ln(z) = ln(|z|) + i * arg(z)
                return math::log(math::abs(argument))
                       + Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)} * math::arg(argument);
            }
        };

        //! The log2 trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Log2::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& argument) const
            {
                return math::log(argument) / math::log(static_cast<T>(2));
            }
        };

        //! The log10 trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Log10::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& argument) const
            {
                return math::log(argument) / math::log(static_cast<T>(10));
            }
        };

        //! The pow trait specialization for complex types.
        template<typename T_MathImpl, typename T, typename U>
        struct Pow::Op<T_MathImpl, Complex<T>, Complex<U>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& base, Complex<U> const& exponent) const
            {
                // Type promotion matching rules of complex std::pow but simplified given our math only supports float
                // and double, no long double.
                using Promoted
                    = Complex<std::conditional_t<isDecayed_v<T, float> && isDecayed_v<U, float>, float, double>>;
                // pow(z1, z2) = e^(z2 * log(z1))
                return math::exp(Promoted{exponent} * math::log(Promoted{base}));
            }
        };

        //! The pow trait specialization for complex and real types.
        template<typename T_MathImpl, typename T, typename U>
        struct Pow::Op<T_MathImpl, Complex<T>, U>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& base, U const& exponent) const
            {
                return math::pow(base, Complex<U>{exponent});
            }
        };

        //! The pow trait specialization for real and complex types.
        template<typename T_MathImpl, typename T, typename U>
        struct Pow::Op<T_MathImpl, T, Complex<U>>
        {
            constexpr auto operator()(T_MathImpl, T const& base, Complex<U> const& exponent) const
            {
                return math::pow(Complex<T>{base}, exponent);
            }
        };

        //! The rsqrt trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Rsqrt::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                return static_cast<T>(1.0) / math::sqrt(arg);
            }
        };

        //! The sin trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Sin::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // sin(z) = (exp(i * z) - exp(-i * z)) / 2i
                return (math::exp(Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)} * arg)
                        - math::exp(Complex<T>{static_cast<T>(0.0), static_cast<T>(-1.0)} * arg))
                       / Complex<T>{static_cast<T>(0.0), static_cast<T>(2.0)};
            }
        };

        //! The sinh trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Sinh::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // sinh(z) = (exp(z) - exp(-i * z)) / 2
                return (math::exp(arg) - math::exp(static_cast<T>(-1.0) * arg)) / static_cast<T>(2.0);
            }
        };

        //! The sincos trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct SinCos::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(
                T_MathImpl,
                Complex<T> const& arg,
                Complex<T>& result_sin,
                Complex<T>& result_cos) const -> void
            {
                result_sin = math::sin(arg);
                result_cos = math::cos(arg);
            }
        };

        //! The sqrt trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Sqrt::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& argument) const
            {
                // Branch cut along the negative real axis (same as for std::complex),
                // principal value of sqrt(z) = sqrt(|z|) * e^(i * arg(z) / 2)
                auto const halfArg = T(0.5) * math::arg(argument);
                auto re = T{}, im = T{};
                math::sincos(halfArg, im, re);
                return math::sqrt(math::abs(argument)) * Complex<T>(re, im);
            }
        };

        //! The tan trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Tan::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // tan(z) = i * (e^-iz - e^iz) / (e^-iz + e^iz) = i * (1 - e^2iz) / (1 + e^2iz)
                // Warning: this straightforward implementation can easily result in NaN as 0/0 or inf/inf.
                auto const expValue = math::exp(Complex<T>{static_cast<T>(0.0), static_cast<T>(2.0)} * arg);
                return Complex<T>{static_cast<T>(0.0), static_cast<T>(1.0)} * (static_cast<T>(1.0) - expValue)
                       / (static_cast<T>(1.0) + expValue);
            }
        };

        //! The tanh trait specialization for complex types.
        template<typename T_MathImpl, typename T>
        struct Tanh::Op<T_MathImpl, Complex<T>>
        {
            constexpr auto operator()(T_MathImpl, Complex<T> const& arg) const
            {
                // tanh(z) = (e^z - e^-z)/(e^z+e^-z)
                return (math::exp(arg) - math::exp(static_cast<T>(-1.0) * arg))
                       / (math::exp(arg) + math::exp(static_cast<T>(-1.0) * arg));
            }
        };

    } // namespace internal

#endif

} // namespace alpaka::math

namespace alpaka::trait
{
    template<typename T>
    struct GetValueType<math::internal::Complex<T>>
    {
        using type = T;
    };
} // namespace alpaka::trait
