# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'leap'
copyright = '2024, Tae Yoon (Harry) Lee, Ainsleigh Hill'
author = 'Tae Yoon (Harry) Lee, Ainsleigh Hill'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.todo',
    'sphinx.ext.coverage',
    # 'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    # 'sphinx.ext.graphviz'
    "sphinx_immaterial",
    "sphinxcontrib.bibtex",
    # "sphinx_immaterial.apidoc.python.apigen"
]


# object_description_options = []
# object_description_options.append(("py:.*", dict(wrap_signatures_with_css=True)))


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
html_title = "LEAP"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']

html_css_files = [
    'css/custom.css'
]

bibtex_bibfiles = ["_static/bibliography.bib"]

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "",
    "repo_url": "https://github.com/resplab/leap/",
    "repo_name": "leap",
    "edit_uri": "blob/main/docs",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
        "content.code.annotate"
    ],
    "palette": [
        {"primary": "cyan"}
    ],
    "version_dropdown": False,
    "toc_title_is_page_title": True,
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/resplab/leap/",
            "name": "Source on github.com",
        }
    ]
}


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints_format = "fully-qualified"
