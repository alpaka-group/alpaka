/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onHost/demangledName.hpp"
#include "alpaka/onHost/logger/lvl.hpp"
#include "alpaka/unused.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <ostream>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>

namespace alpaka::onHost::logger::internal
{
    /** Write all output to std::cerr
     *
     * The output is not buffered and will be written immediately, it is **NOT** threadsafe.
     *
     * @todo separate the indention level from the writer
     * @todo write additional logger, std::cout and thread save logger
     */
    struct StdErr
    {
        static StdErr& get()
        {
            static StdErr inst = StdErr{};
            return inst;
        }

        std::ostream& operator<<(auto const& input) const
        {
            return std::cerr << input;
        }

        /** increase the indention level
         *
         * @return the indention level for the current message
         */
        int enter()
        {
            return m_indentLvl++;
        }

        /** decrease the indention level
         *
         * @return the indention level for the current message
         */
        int leave()
        {
            return --m_indentLvl;
        }

        /** current indention level
         *
         * @return the indention level for the current message
         */
        int current()
        {
            return m_indentLvl.load();
        }

    private:
        std::atomic<int> m_indentLvl = 1;
    };

    /** Indent the message if needed and forward it to the output writer
     *
     * If input is indented depends on the preprocessor define ALPAKA_LOG_INDENT
     */
    inline void indent(auto& writer, [[maybe_unused]] int indentLvl)
    {
#if defined(ALPAKA_LOG_INDENT)
        for(int i = 0; i < indentLvl; ++i)
            i == 0 ? (writer << "|-") : (writer << "--");
        if(indentLvl)
#endif
            writer << " ";
    }

    /** Adjust the length of a string to a minimum length
     *
     * @param str input string
     * @param n minimum number of characters, if the string is shorter than this number, it will be padded with a
     * padding character
     * @return new string with a  minimum number of characters
     */
    inline std::string adjStringLength(std::string str, size_t n, char const paddingCharacter = ' ')
    {
        if(str.length() >= n)
        {
            return str;
        }
        str.resize(n, paddingCharacter);
        return str;
    }

    /** shortening the function signatures to become human-readable
     *
     * If the name is simplified depends on the preprocessor define ALPAKA_LOG_DETAIL_SHORT
     */
    inline std::string adjDetails(std::string const& str)
    {
#if defined(ALPAKA_LOG_DETAIL_SHORT)
        return onHost::simplifyFunctionSignature(str);
#else
        return str;
#endif
    }

    /** Log the entry and exit of a scope */
    template<logger::concepts::Level T_LogLvl, typename T_Writer = StdErr>
    struct Scoped
    {
    public:
        Scoped(T_LogLvl logLvl, std::source_location const& location)
            : m_functionName{adjDetails(location.function_name())}
            , m_prefix{std::string("[") + adjStringLength(logLvl.getName(), 6) + "]"}
            , m_startTime{std::chrono::high_resolution_clock::now()}
            , m_writer{T_Writer::get()}
        {
            m_writer << m_prefix << "[+]";
            indent(m_writer, m_writer.enter());
            m_writer << m_functionName << std::endl;
        }

        Scoped(T_LogLvl logLvl) : m_writer{T_Writer::get()}, m_enableOutput{false}
        {
            alpaka::unused(logLvl);
        }

        Scoped(Scoped const&) = delete;
        Scoped(Scoped&&) = delete;
        Scoped& operator=(Scoped const&) = delete;
        Scoped& operator=(Scoped&&) = delete;

        ~Scoped()
        {
            if(m_enableOutput)
            {
                auto const endTime = std::chrono::high_resolution_clock::now();
                double durationInSeconds = std::chrono::duration<double, std::milli>(endTime - m_startTime).count();

                m_writer << m_prefix << "[-]";
                indent(m_writer, m_writer.leave());
                m_writer << m_functionName << " " << durationInSeconds << " ms" << std::endl;
            }
        }

    private:
        std::string m_functionName;
        std::string m_prefix;
        decltype(std::chrono::high_resolution_clock::now()) m_startTime;
        T_Writer& m_writer;
        bool m_enableOutput = true;
    };

    /** Write a meta data message to the output
     *
     * @tparam T_Callable callable without arguments which provides a string which should be written to the output
     */
    template<logger::concepts::Level T_LogLvl, typename T_Callable, typename T_Writer = StdErr>
    requires(std::is_invocable_r_v<std::string, T_Callable>)
    struct Info
    {
    public:
        Info(T_LogLvl logLvl, T_Callable const& callable, std::source_location const& location)
        {
            auto fullPrefix = std::string("[") + adjStringLength(logLvl.getName(), 6) + "]";

            auto& writer = T_Writer::get();
            std::stringstream ss;
            ss << "   ";
            writer << fullPrefix << ss.str();
            indent(writer, writer.current());
            writer << callable() << " " << adjDetails(location.function_name()) << " " << location.file_name() << ":"
                   << location.line() << std::endl;
        }

        Info(Info const&) = delete;
        Info(Info&&) = delete;
        Info& operator=(Info const&) = delete;
        Info& operator=(Info&&) = delete;

        ~Info() = default;
    };
} // namespace alpaka::onHost::logger::internal
