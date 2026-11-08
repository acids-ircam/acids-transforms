"""Sphinx configuration for acids_transforms (built locally with `make -C docs html`, and by Read the Docs)."""
import os
import sys
from importlib import metadata

# make the package importable without installing it (Read the Docs installs it anyway, see .readthedocs.yaml)
sys.path.insert(0, os.path.abspath(".."))

project = "acids_transforms"
author = "Axel Chemla--Romeu-Santos"
copyright = "2022-2026, Axel Chemla--Romeu-Santos"
try:
    release = metadata.version("acids_transforms")
except metadata.PackageNotFoundError:
    release = "0.1.4"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (markdown) -----------------------------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "fieldlist"]
myst_heading_anchors = 3

# -- autodoc -------------------------------------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    # nn.Module internals and test helpers are not part of the public API
    "exclude-members": "test_forward, test_inversion, test_scripted_transform, training, dump_patches, "
                       "call_super_init, T_destination",
}
autodoc_typehints = "description"
# do not show docstrings inherited from torch.nn.Module (forward, etc.)
autodoc_inherit_docstrings = False
autodoc_class_signature = "separated"
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "torchaudio": ("https://docs.pytorch.org/audio/stable", None),
}

# -- HTML ----------------------------------------------------------------------------------------------------------
html_theme = "furo"
html_title = f"acids_transforms {release}"
# only if present : git does not track empty directories
html_static_path = ["_static"] if os.path.isdir(os.path.join(os.path.dirname(__file__), "_static")) else []
html_theme_options = {
    "source_repository": "https://github.com/domkirke/acids_transforms/",
    "source_branch": "master",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
