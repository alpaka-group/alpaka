# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.

import os
import sys

# load extended command usable in the rst documents e.g. filteredliteralinclude
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "sphinx_extensions")))

# allows to import module `build_helper`
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sphinx_helper.cheatsheet import generate_cheatsheet
from sphinx_helper.single_header import generate_single_header
from sphinx_helper.utils import on_rtd
from sphinx_helper.doxygen import generate_doxygen
from sphinx_helper.math_function_families import generate_math_function_families


def setup(app):
    # Doxygen XML must exist before Sphinx/Breathe reads the documents.
    app.connect("builder-inited", generate_cheatsheet)
    app.connect("builder-inited", generate_doxygen)
    app.connect("builder-inited", generate_math_function_families)
    app.connect("build-finished", generate_single_header)


# -- Project information -----------------------------------------------------

project = "alpaka"
copyright = "Documentation under CC-BY 4.0"
author = "The alpaka team."
# The short X.Y version.
version = "3.1.0"
# The full version, including alpha/beta/rc tags.
release = "3.1.0-dev"

# The master toctree document.
master_doc = "index"

# -- General configuration ---------------------------------------------------

highlight_language = "c++"

show_authors = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    #    'sphinx.ext.napoleon',
    "filtered_literalinclude",
    "sphinx_rtd_theme",
    "sphinxcontrib.programoutput",
    #    'matplotlib.sphinxext.plot_directive'
    #    "sphinx.ext.autosectionlabel",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
]

source_suffix = [".rst"]
master_doc = "index"
language = "en"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"  # 'default'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The cheatsheet PDF is generated directly into the HTML output directory by
# sphinx_helper/cheatsheet.py (basic/cheatsheet.html links to ../cheatsheet.pdf).

# modifies the HTML Sphinx Doc layout
html_css_files = ["custom.css"]

html_logo = "../logo/alpaka.svg"
html_theme_options = {
    "logo_only": True,
    "collapse_navigation": False,
    "navigation_depth": 2,
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "alpakadoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    "preamble": r"\setcounter{tocdepth}{2}",
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
latex_logo = "../logo/alpaka.png"

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "alpaka-doc.tex", "alpaka Documentation", "The alpaka Community", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "alpaka", "alpaka Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "alpaka",
        "alpaka Documentation",
        author,
        "alpaka",
        "Abstraction Library for Parallel Kernel Acceleration",
        """
     The alpaka library is a header-only C++20 abstraction library for
     accelerator development. Its aim is to provide performance portability
     across accelerators through the abstraction (not hiding!) of the underlying
     levels of parallelism.
     """,
    ),
]

# -- Options for Epub output -------------------------------------------------

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# -- processing --

if on_rtd():
    pass
else:
    from sphinx.util import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "single header build can be force build or skipped with the environment variable 'ALPAKA_DOC_SINGLE_HEADER=0|OFF|1|ON'"
    )
    logger.info(
        "doxygen build can be force build or skipped with the environment variable 'ALPAKA_DOC_DOXYGEN=0|OFF|1|ON'"
    )
    logger.info(
        "cheatsheet.pdf build can be force build or skipped with the environment variable 'ALPAKA_DOC_CHEATSHEET=0|OFF|1|ON'"
    )
