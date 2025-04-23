import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = 'SwarmS.py'
copyright = '2025, Stefano Covone, Italo Napolitano, Davide Salzano'
author = 'Stefano Covone, Italo Napolitano, Davide Salzano'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# Ensure Sphinx includes class attributes
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',  # Ensures attributes appear in the correct order
}

autodoc_typehints = "description"  # Ensures type hints appear correctly

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_custom_sections = [('Config requirements', 'params_style')]


# Exclude unnecessary files
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    '_templates'
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navigation_depth": 4,
    "show_nav_level": 4,
    "collapse_navigation": False,
}
html_static_path = ['_static']

# Set master doc (for older Sphinx versions)
master_doc = "index"



