.. _dev-logging:

Logging
=======

.. sectionauthor:: René Widera

Sometimes it is useful to get an overview about function call and additional low level information.
Logging gives you a lightweight overview with a reduced callstack and meta information.
Mostly leaf operations which call native backends e.g. CUDA, HIP, and SYCL are instrumented.
Via logging channels the scope of information can be filtered.

.. _logging-level:

Logging Level
-------------

Via logging level the scope of information can be filtered.

  * ``Device=1`` - logs device information
  * ``Event=2`` - logs event information
  * ``Memory=4`` - logs memory information
  * ``Queue=8`` - logs queue information
  * ``Kernel=16`` - logs kernel information

CMake Logging Options
---------------------

* ``alpaka_LOG=<X>`` where ``X`` can be

   - ``OFF`` - to disable logging, other CMake option ``alpaka_LOG_*`` will not be available.
   - ``static`` - will enable compile time logging where the logging level can be selected in CMake, additional CMake option ``alpaka_LOG_STATIC_*`` will be available.
   - ``dynamic`` - will enable runtime logging where the logging level can be selected via the environment variable ``ALPAKA_LOG_DYNAMIC_LVL=<lvl>``

     - ``lvl`` - is the number of :ref:`logging-level` or the sum of multiple lvl to select multiple levels within the same output

  If logging is off there will be no runtime overhead introduced.

   - ``static`` - will introduce small overhead for message formatting and the output.
     If a logging level is disabled these messages will be removed at compile time and no runtime overhead is introduced.
   - ``dynamic``  - will introduce overhead for message formatting and additional small overhead for a runtime lookup (one if condition).

* ``alpaka_LOG_FUNCTIONS=<X>`` - where ``X`` can be ``ON`` or ``OFF`` create a reduced call stack and show the function entry and exit. The exit will show the time in milliseconds between the entry and exit of the function.

  .. code:: c++

     [Memory][+] void internal::generic::fill(auto:463&, auto:464, auto:465&&, T_Value) [with T_Value = unsigned int; auto:463 = onHost::cpu::Queue<...> > >; auto:464 = exec::CpuOmpBlocks; auto:465 = alpaka::View<...> >, alpaka::Alignment<...> >]
     [Memory][-] void internal::generic::fill(auto:463&, auto:464, auto:465&&, T_Value) [with T_Value = unsigned int; auto:463 = onHost::cpu::Queue<...> > >; auto:464 = exec::CpuOmpBlocks; auto:465 = alpaka::View<...> >, alpaka::Alignment<...> >] 0.247257 ms

* ``alpaka_LOG_INFO=<X>`` - where ``X`` can be ``ON` or ``OFF`` provides meta information to different functions and objects.
  The element of the output is the absolute path and line number of the file where the logging message is coming from

  .. code:: c++

     [Memory]    SharedBuffer{ dim=1, api= Host, extents={123456}, pitches={4} , alignment=16 } auto onHost::internal::Alloc::Op<...>, T_Extents>::operator()(onHost::cpu::Device<...>&, const T_Extents&) const [with T_Type = unsigned int; T_Platform = onHost::cpu::Platform<...>; T_Extents = alpaka::Vec<...> >] alpaka/api/host/Device.hpp:198

* ``alpaka_LOG_DETAIL=<X>`` - where ``X``  can be ``short`` or ``long`` which will format the function name details.

  .. code:: c++

     // short signature
     [Memory]    SharedBuffer{ dim=1, api= Host, extents={123456}, pitches={4} , alignment=16 } auto onHost::internal::Alloc::Op<...>, T_Extents>::operator()(onHost::cpu::Device<...>&, const T_Extents&) const [with T_Type = unsigned int; T_Platform = onHost::cpu::Platform<...>; T_Extents = alpaka::Vec<...> >] alpaka/api/host/Device.hpp:198
     // long signature
     [Memory]    SharedBuffer{ dim=1, api= Host, extents={123456}, pitches={4} , alignment=16 } auto alpaka::onHost::internal::Alloc::Op<T_Type, alpaka::onHost::cpu::Device<T_Platform>, T_Extents>::operator()(alpaka::onHost::cpu::Device<T_Platform>&, const T_Extents&) const [with T_Type = unsigned int; T_Platform = alpaka::onHost::cpu::Platform<alpaka::deviceKind::Cpu>; T_Extents = alpaka::Vec<long unsigned int, 1, alpaka::ArrayStorage<long unsigned int, 1> >] alpaka/api/host/Device.hpp:198


* ``alpaka_LOG_INDENT=<X>`` - where ``X``  can be ``ON`` or ``OFF`` which will format the output from ``alpaka_LOG_FUNCTIONS`` and ``alpaka_LOG_INFO`` depending on the call stack depth.

  .. code:: c++

     [Queue ][+]|- void onHost::internal::Fill::Op<...>, T_Dest, T_Value, T_Extents>::operator()(onHost::cpu::Queue<...>&, auto:515&&, T_Value, const T_Extents&) const requires (same_as<...>, T_Dest, T_Value, T_Extents>::operator()::dest)>::type, T_Dest>) && (same_as<...>, T_Dest, T_Value, T_Extents>::operator()::dest)>::type>::type, T_Value>) [with auto:515 = onHost::SharedBuffer<...> >, alpaka::Alignment<...> >&; T_Device = onHost::cpu::Device<...> >; T_Dest = onHost::SharedBuffer<...> >, alpaka::Alignment<...> >; T_Value = unsigned int; T_Extents = alpaka::Vec<...> >]
     [Memory][+]|--- void internal::generic::fill(auto:463&, auto:464, auto:465&&, T_Value) [with T_Value = unsigned int; auto:463 = onHost::cpu::Queue<...> > >; auto:464 = exec::CpuOmpBlocks; auto:465 = alpaka::View<...> >, alpaka::Alignment<...> >]
     [Memory]   |----- fill{ extents={123456}, dst=View{ dim=1, api= Host, extents={123456}, pitches={4} , alignment=16 }, value_type=unsigned int, FrameSpec{ frames={15}, frameExtent={512}, ThreadSpec{ blocks={15}, threads={512} } } } void internal::generic::fill(auto:463&, auto:464, auto:465&&, T_Value) [with T_Value = unsigned int; auto:463 = onHost::cpu::Queue<...> > >; auto:464 = exec::CpuOmpBlocks; auto:465 = alpaka::View<...> >, alpaka::Alignment<...> >] /home/widera/workspace/alpaka2/include/alpaka/api/generic.hpp:63
     [Memory][-]|--- void internal::generic::fill(auto:463&, auto:464, auto:465&&, T_Value) [with T_Value = unsigned int; auto:463 = onHost::cpu::Queue<...> > >; auto:464 = exec::CpuOmpBlocks; auto:465 = alpaka::View<...> >, alpaka::Alignment<...> >] 0.942437 ms
     [Queue ][-]|- void onHost::internal::Fill::Op<...>, T_Dest, T_Value, T_Extents>::operator()(onHost::cpu::Queue<...>&, auto:515&&, T_Value, const T_Extents&) const requires (same_as<...>, T_Dest, T_Value, T_Extents>::operator()::dest)>::type, T_Dest>) && (same_as<...>, T_Dest, T_Value, T_Extents>::operator()::dest)>::type>::type, T_Value>) [with auto:515 = onHost::SharedBuffer<...> >, alpaka::Alignment<...> >&; T_Device = onHost::cpu::Device<...> >; T_Dest = onHost::SharedBuffer<...> >, alpaka::Alignment<...> >; T_Value = unsigned int; T_Extents = alpaka::Vec<...> >] 1.12421 ms

The following CMake options requires ``alpaka_LOG=static``

* ``alpaka_LOG_STATIC_Device`` - activate logging for static device information
* ``alpaka_LOG_STATIC_Event`` - activate logging for static event information
* ``alpaka_LOG_STATIC_Memory`` - activate logging for static memory information
* ``alpaka_LOG_STATIC_Queue`` - activate logging for static queue information
* ``alpaka_LOG_STATIC_Kernel`` - activate logging for static kernel information


C++ Code logging
----------------

* Logger level are required for both following calls:
  :cpp:class:`alpaka::onHost::logger::Device`
  :cpp:class:`alpaka::onHost::logger::Event`
  :cpp:class:`alpaka::onHost::logger::Memory`
  :cpp:class:`alpaka::onHost::logger::Queue`
  :cpp:class:`alpaka::onHost::logger::Kernel`

* The logger function macro marking the scope of a function, ``ALPAKA_LOG_FUNCTION()`` must be on the very beginning of the function.
  The entry is the macro function call and the exit is logged when the current scope is ending.
  This means ``ALPAKA_LOG_FUNCTION()`` is scope aware.

  .. code:: c++

     void foo()
     {
         ALPAKA_LOG_FUNCTION(onHost::logger::device);
         // very complex code
     }

* The logger function macro to output any kind of information, ``ALPAKA_LOG_INFO()`` requires a lvl and a lambda function.
  The lambda is only called in case the logging level is selected. Capturing variable by reference preferred to keep the runtime overhead small.

  .. code:: c++

     void foo(auto* ptr, auto extents)
     {
         ALPAKA_LOG_INFO(
            onHost::logger::memory,
            [&]()
            {
                std::stringstream ss;
                ss << "{extents=" << extents << ", ptr=" << ptr}";
                return ss.str();
            });
     }

Logging levels can be combined via the plus operator. ``onHost::logger::memory + onHost::logger::device``.
In this case the combined output level suffix will always show the first name even if this level is deactivated, this should keep the output readable.
