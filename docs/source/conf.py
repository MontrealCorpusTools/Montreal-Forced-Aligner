#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Montreal Forced Aligner documentation build configuration file, created by
# sphinx-quickstart on Wed Jun 15 13:27:38 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../../"))
import montreal_forced_aligner  # noqa

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.4'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "external_links",
    # "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx_panels",
    "sphinx.ext.viewcode",
    "sphinxcontrib.autoprogram",
    "sphinxemoji.sphinxemoji",
    # "sphinx_autodoc_typehints",
]
panels_add_bootstrap_css = False
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

extlinks = {
    "mfa_pr": ("https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/pull/%s", "PR #%s"),
}

xref_links = {
    "mfa_mailing_list": ("MFA mailing list", "https://groups.google.com/g/mfa-users"),
    "mfa_github": ("MFA GitHub Repo", "https://groups.google.com/g/mfa-users"),
    "mfa_github_issues": (
        "MFA GitHub Issues",
        "https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues",
    ),
    "memcauliffe.com": ("Michael McAuliffe's blog", "https://memcauliffe.com"),
    "@wavable": ("@wavable", "https://twitter.com/wavable"),
    "sonderegger": ("Morgan Sonderegger", "http://people.linguistics.mcgill.ca/~morgan/"),
    "wagner": ("Michael Wagner", "https://prosodylab.org/"),
    "coles": ("Arlie Coles", "https://a-coles.github.io/"),
    "stengel-eskin": ("Elias Stengel-Eskin", "https://esteng.github.io/"),
    "socolof": ("Michaela Socolof", "https://mcqll.org/people/socolof.michaela/"),
    "mihuc": ("Sarah Mihuc", "https://www.cs.mcgill.ca/~smihuc/"),
    "wsl": ("Windows Subsystem for Linux", "https://docs.microsoft.com/en-us/windows/wsl/install"),
    "kaldi": ("Kaldi", "http://kaldi-asr.org/"),
    "kaldi_github": ("Kaldi GitHub", "https://github.com/kaldi-asr/kaldi"),
    "htk": ("HTK", "http://htk.eng.cam.ac.uk/"),
    "phonetisaurus": ("Phonetisaurus", "https://github.com/AdolfVonKleist/Phonetisaurus"),
    "pynini": ("Pynini", "https://www.openfst.org/twiki/bin/view/GRM/Pynini"),
    "prosodylab_aligner": ("Prosodylab-aligner", "http://prosodylab.org/tools/aligner/"),
    "p2fa": (
        "Penn Phonetics Forced Aligner",
        "https://www.ling.upenn.edu/phonetics/old_website_2015/p2fa/",
    ),
    "fave": ("FAVE-align", "https://github.com/JoFrhwld/FAVE/wiki/FAVE-align"),
    "maus": ("MAUS", "http://www.bas.uni-muenchen.de/Bas/BasMAUS.html"),
    "praat": ("Praat", "http://www.fon.hum.uva.nl/praat/"),
    "easy_align": ("EasyAlign", "http://latlcui.unige.ch/phonetique/easyalign.php"),
    "gentle": ("Gentle", "https://lowerquality.com/gentle/"),
    "chodroff_kaldi": ("Kaldi tutorial", "https://eleanorchodroff.com/tutorial/kaldi/index.html"),
    "chodroff_phonetics": (
        "Corpus Phonetics Tutorial",
        "https://eleanorchodroff.com/tutorial/intro.html",
    ),
    "coqui": ("Coqui", "https://coqui.ai/"),
    "conda_installation": (
        "Conda installation",
        "https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html",
    ),
    "conda_forge": ("Conda Forge", "https://conda-forge.org/"),
    "pydata_sphinx_theme": (
        "Pydata Sphinx Theme",
        "https://pydata-sphinx-theme.readthedocs.io/en/latest/",
    ),
    "mfa_reorg_scripts": (
        "MFA-reorganization-scripts repository",
        "https://github.com/MontrealCorpusTools/MFA-reorganization-scripts",
    ),
}

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------


autosummary_generate = True
autodoc_typehints = "none"
# autodoc_typehints_description_target = 'documented'
# autoclass_content = 'both'
autodoc_docstring_signature = True
autodoc_type_aliases = {
    "MultispeakerDictionary": "montreal_forced_aligner.dictionary.MultispeakerDictionary",
    "Trainer": "montreal_forced_aligner.abc.Trainer",
    "Aligner": "montreal_forced_aligner.abc.Aligner",
    "DictionaryData": "montreal_forced_aligner.dictionary.DictionaryData",
    "Utterance": "montreal_forced_aligner.corpus.Utterance",
    "File": "montreal_forced_aligner.corpus.File",
    "FeatureConfig": "montreal_forced_aligner.config.FeatureConfig",
    "multiprocessing.context.Process": "multiprocessing.Process",
    "mp.Process": "multiprocessing.Process",
    "Speaker": "montreal_forced_aligner.corpus.Speaker",
}

napoleon_preprocess_types = False
napoleon_attr_annotations = False
napoleon_use_param = True
napoleon_type_aliases = {
    "Labels": "List[str]",
}
typehints_fully_qualified = False
# numpydoc_xref_param_type = True
# numpydoc_show_inherited_class_members = False
numpydoc_show_class_members = False
# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Montreal Forced Aligner"
copyright = f"2018-{date.today().year}, Montreal Corpus Tools"
author = "Montreal Corpus Tools"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = ".".join(montreal_forced_aligner.utils.get_mfa_version().split(".", maxsplit=2)[:2])
# The full version, including alpha/beta/rc tags.
release = montreal_forced_aligner.utils.get_mfa_version()

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#
# today = ''
#
# Else, today_fmt is used as the format for a strftime call.
#
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all
# documents.
#
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
#
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#
# show_authors = False

# nitpicky = True
nitpick_ignore = [("py:class", "optional")]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_logo = "_static/logo_long.svg"
html_favicon = "_static/favicon.ico"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner",
            "icon": "fab fa-github-square",
        },
    ],
    "google_analytics_id": "UA-73068199-4",
    "show_nav_level": 1,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "collapse_navigation": False,
}
html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise interprise
    "github_user": "MontrealCorpusTools",
    "github_repo": "Montreal-Forced-Aligner",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
#
# html_title = 'Montreal Forced Aligner v1.0'

# A shorter title for the navigation bar.  Default is the same as html_title.
#
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
# html_logo = None

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/style.css",
]

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#
# html_extra_path = []

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
#
# html_last_updated_fmt = None

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#
# html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'], }
html_sidebars = {"**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]}
# Additional templates that should be rendered to pages, maps page names to
# template names.
#
# html_additional_pages = {}

# If false, no module index is generated.
#
# html_domain_indices = True

# If false, no index is generated.
#
# html_use_index = True

# If true, the index is split into individual pages for each letter.
#
# html_split_index = False

# If true, links to the reST sources are added to the pages.
#
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'h', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'r', 'sv', 'tr', 'zh'
#
# html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# 'ja' uses this config value.
# 'zh' user can custom change `jieba` dictionary path.
#
# html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#
# html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = "MontrealForcedAlignerdoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "MontrealForcedAligner.tex",
        "Montreal Forced Aligner Documentation",
        "Montreal Corpus Tools",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#
# latex_use_parts = False

# If true, show page references after internal links.
#
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
#
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
#
# latex_appendices = []

# If false, no module index is generated.
#
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "montrealforcedaligner", "Montreal Forced Aligner Documentation", [author], 1)
]

# If true, show URL addresses after external links.
#
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "MontrealForcedAligner",
        "Montreal Forced Aligner Documentation",
        author,
        "MontrealForcedAligner",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
#
# texinfo_appendices = []

# If false, no module index is generated.
#
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#
# texinfo_no_detailmenu = False
