# Configuration file for the Sphinx documentation builder.
import os
import shutil
import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup

__location__ = Path(os.path.dirname(__file__))
__src__ = __location__ / "../.."

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, str(__src__))


# -- Project information

project = "ESGPTTaskQuerying"
copyright = "2024, Justin Xu & Matthew McDermott"
author = "Justin Xu & Matthew McDermott"

release = "0.0.1"
version = "0.0.1"


def ensure_pandoc_installed(_):
    """Source: https://stackoverflow.com/questions/62398231/building-docs-fails-due-to-missing-pandoc"""
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = str(__location__ / "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir

    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


# -- Run sphinx-apidoc -------------------------------------------------------
# This ensures we don't need to run apidoc manually.

# TODO: use https://github.com/sphinx-extensions2/sphinx-autodoc2

from sphinx.ext import apidoc

output_dir = __location__ / "api"
module_dir = __src__ / "src/esgpt_task_querying"
if output_dir.is_dir():
    shutil.rmtree(output_dir)

try:
    cmd_line = f"--implicit-namespaces -e -f -o {output_dir} {module_dir}"
    apidoc.main(cmd_line.split(" "))
except Exception as e:  # pylint: disable=broad-except
    print(f"Running `sphinx-apidoc {cmd_line}` failed!\n{e}")


# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.collections",
    "sphinx_subfigure",
    "sphinx_immaterial",
    "myst_parser",
    "nbsphinx",
]

collections_dir = __location__ / "_collections"
if not collections_dir.is_dir():
    os.mkdir(collections_dir)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Control options for included jupyter notebooks.
nb_execution_mode = "off"

# -- Options for HTML output

html_theme = "sphinx_immaterial"


# -- Options for EPUB output
epub_show_urls = "footnote"
