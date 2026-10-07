Documentation
=============

.. sectionauthor:: Simeon Ehrig, Axel Huebl

In the following section, we explain how you can contribute to this documentation.

The documentation consists of two main parts. The first part is a manual-styled documentation that contains text blocks, images, and other media. The second part is the API documentation.

Documentation Kinds
-------------------

Manuel Documentation
````````````````````

We use `Sphinx Doc <https://www.sphinx-doc.org/en/master/index.html>`_  for manual-style documentation and host the rendered documentation at https://readthedocs.com.

If you are reading the `HTML version <https://alpaka.readthedocs.io>`_ and want to improve or correct existing pages, check the "*Edit on GitHub*" link on the right upper corner of each document.

Alternatively, go to `docs/source` in our source code and follow the directory structure of `reStructuredText`_ (``.rst``) files there.
For intrusive changes, like structural changes to chapters, please open an issue to discuss them beforehand.

.. _reStructuredText: https://www.sphinx-doc.org/en/stable/rest.html

API documentation
`````````````````

The API documentation is generated in the source code using Doxygen via source code comments. The Doxygen documentation is automatically generated as part of the Sphinx documentation.

If you would like to improve the API documentation, please fork the project, improve the documentation, and open a pull request.

Preview
-------

As part of a pull request, the manual and API documentation are created and made available on a temporary website. The preview can be accessed via the CI job link.

.. image:: /images/pull_request_doc_ci_job.png

Build Locally
-------------

Sphinx Doc is a Python software. Therefore, it is strongly recommended that you create a Python environment to generate the documentation.

The following example shows how to set up a virtual Python environment for Sphinx Doc and generate the documentation.

Install all dependencies:

.. code-block:: bash

    python3 -m venv alpaka-doc-env
    # activate environment
    source alpaka-doc-env/bin/activate
    cd docs
    # install dependencies
    pip3 install -r requirements.txt
    # install doxygen
    curl https://www.doxygen.nl/files/doxygen-1.16.0.linux.bin.tar.gz -o /tmp/doxygen.tar.gz
    tar -xf /tmp/doxygen.tar.gz -C /tmp/
    cp /tmp/doxygen-1.16.0/bin/doxygen $VIRTUAL_ENV/bin

Build documentation:

.. code-block:: bash

    # activate environment
    source alpaka-doc-env/bin/activate
    cd docs
    # build documentation
    make html
    # chromium and other browser also works
    firefox ./build/html/index.html


Useful Links
------------

 * `A primer on writing reStructuredText files for sphinx <https://www.sphinx-doc.org/en/stable/rest.html>`_
 * `reStructuredText vs. Markdown <https://eli.thegreenplace.net/2017/restructuredtext-vs-markdown-for-technical-documentation/>`_
 * `readthedocs on github <https://github.com/readthedocs/readthedocs.org>`_
