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

project = "ACES: Automatic Cohort Extraction System for Event-Streams"
copyright = "2024, Justin Xu & Matthew McDermott"
author = "Justin Xu & Matthew McDermott"

release = "0.2.5"
version = "0.2.5"


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


# -- Run sphinx-apidoc
# This ensures we don't need to run apidoc manually.

# TODO: use https://github.com/sphinx-extensions2/sphinx-autodoc2

from sphinx.ext import apidoc

output_dir = __location__ / "api"
module_dir = __src__ / "src/aces"
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
    "sphinx.ext.imgconverter",
    "sphinxcontrib.collections",
    "sphinx_subfigure",
    "sphinx_immaterial",
    "myst_parser",
    "nbsphinx",
]

nbsphinx_allow_errors = True


collections_dir = __location__ / "_collections"
if not collections_dir.is_dir():
    os.mkdir(collections_dir)

python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/" + python_version, None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pandera": ("https://pandera.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "setuptools": ("https://setuptools.pypa.io/en/stable/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
    "hyperimpute": ("https://hyperimpute.readthedocs.io/en/latest/", None),
    "xgbse": ("https://loft-br.github.io/xgboost-survival-embeddings/", None),
    "lifelines": ("https://lifelines.readthedocs.io/en/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
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

# Configure MyST-Parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_update_mathjax = True

# MyST URL schemes.
myst_url_schemes = {
    "http": None,
    "https": None,
    "ftp": None,
    "mailto": None,
    "repo-code": "https://github.com/justin13601/ACES/tree/main/{{path}}#{{fragment}}",
    # "doi": "https://doi.org/{{path}}",
    # "gh-issue": {
    #     "url": "https://github.com/executablebooks/MyST-Parser/issue/{{path}}#{{fragment}}",
    #     "title": "Issue #{{path}}",
    #     "classes": ["github"],
    # },
}

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "py:obj"

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
# https://pygments.org/styles/
pygments_style = "tango"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_immaterial"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "ACES",
    # Icon,
    # "logo_icon": "query-512.png",
    # Set you GA account ID to enable tracking
    # "google_analytics_account": "UA-XXXXX",
    #
    # TODO: Sitemap.
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # 'base_url': 'https://project.github.io/project',
    #
    # Set the color and the accent color
    "palette": {"primary": "green", "accent": "green"},
    # {
    #     "media": "(prefers-color-scheme: light)",
    #     "scheme": "default",
    #     "toggle": {
    #         "icon": "material/toggle-switch-off-outline",
    #         "name": "Switch to dark mode",
    #     },
    # },
    # {
    #     "media": "(prefers-color-scheme: dark)",
    #     "scheme": "slate",
    #     "toggle": {
    #         "icon": "material/toggle-switch",
    #         "name": "Switch to light mode",
    #     },
    # },
    # "color_primary": "green",
    # "color_accent": "green",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/justin13601/ACES",
    "repo_name": "ACES",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 3,
    # If False, expand all TOC entries
    "globaltoc_collapse": True,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"ACES v{version} Documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "ACES Documentation"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "assets/aces_logo_black.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "assets/aces_logo_green.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "aces-doc"


# -- Options for LaTeX output
# latex_engine = "xelatex"
latex_elements = {  # type: ignore
    # The paper size ("letterpaper" or "a4paper").
    "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": "\n".join(
        [
            r"\usepackage{svg}",
            r"\DeclareUnicodeCharacter{2501}{-}",
            r"\DeclareUnicodeCharacter{2503}{|}",
            r"\DeclareUnicodeCharacter{2500}{-}",
            r"\DeclareUnicodeCharacter{2550}{-}",
            r"\DeclareUnicodeCharacter{2517}{+}",
            r"\DeclareUnicodeCharacter{2518}{+}",
            r"\DeclareUnicodeCharacter{2534}{+}",
            r"\DeclareUnicodeCharacter{250C}{+}",
            r"\DeclareUnicodeCharacter{252C}{+}",
            r"\DeclareUnicodeCharacter{2510}{+}",
            r"\DeclareUnicodeCharacter{2502}{|}",
            r"\DeclareUnicodeCharacter{2506}{|}",
            r"\DeclareUnicodeCharacter{2561}{|}",
            r"\DeclareUnicodeCharacter{256A}{|}",
            r"\DeclareUnicodeCharacter{2523}{|}",
            r"\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}",
            r"\DeclareUnicodeCharacter{255E}{|}",
            r"\DeclareUnicodeCharacter{255F}{+}",
            r"\DeclareUnicodeCharacter{254E}{|}",
            r"\DeclareUnicodeCharacter{257C}{-}",
            r"\DeclareUnicodeCharacter{257E}{-}",
            r"\DeclareUnicodeCharacter{2559}{+}",
        ]
    ),
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "aces_documentation.tex", "ACES Documentation", r"Justin Xu \& Matthew McDermott", "manual")
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = ""

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True

# -- Options for EPUB output
epub_show_urls = "footnote"

print(f"loading configurations for {project} {version} ...", file=sys.stderr)


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)
