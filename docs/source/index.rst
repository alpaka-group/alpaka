:orphan:

.. only:: html

  .. image:: ../logo/alpaka.svg

.. only:: latex

  .. image:: ../logo/alpaka.pdf

*alpaka - An Abstraction Library for Parallel Kernel Acceleration*

The alpaka library is a header-only C++14 abstraction library for accelerator development. Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

.. CAUTION::
   The readthedocs pages are work in progress and contain outdated sections.

alpaka - How to Read This Document
----------------------------------

Generally, **follow the manual pages in-order** to get started.
Individual chapters are based on the information of the chapters before.

.. only:: html

   The online version of this document is **versioned** and shows by default the manual of the last *stable* version of alpaka.
   If you are looking for the latest *development* version, `click here <https://alpaka.readthedocs.io/en/latest/>`_.

.. note::

   Are you looking for our latest Doxygen docs for the API?

   - See https://alpaka-group.github.io/alpaka/


.. toctree::
   :caption: BASIC
   :maxdepth: 1

   basic/intro.rst
   basic/install.rst
   basic/abstraction.rst
   basic/library.rst
   basic/cmake_example.rst
   basic/cheatsheet.rst
	      
.. toctree::
   :caption: ADVANCED
   :maxdepth: 1

   advanced/rationale.rst
   advanced/mapping.rst
   advanced/backends.rst
   advanced/details.rst
      
.. toctree::
   :caption: EXTRA INFO
   :maxdepth: 1

   info/similar_projects.rst
	      
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1

   dev/style
   dev/sphinx
   API Reference <https://alpaka-group.github.io/alpaka>

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

