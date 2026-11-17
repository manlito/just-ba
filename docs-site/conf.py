# Sphinx configuration of the documentation site (document 08, D119).
# Build: see docs-site/README.md. The generated pages are written on every build.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gen"))

import just_ba  # noqa: E402  (from a build on PYTHONPATH, or an installed wheel)
import generate  # noqa: E402

generate.main(os.path.join(HERE, "_generated"))

project = "just-ba"
author = "the just-ba authors"
copyright = "the just-ba authors, Apache-2.0"
release = just_ba.__version__
version = release

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", ".venv", "README.md", "gen", "_theme"]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
# The docstring examples run with `sphinx-build -b doctest` (document 08).
doctest_global_setup = "import numpy as np\nimport just_ba\njba = just_ba"

# The live demo (D132): its built page and module, from a WebAssembly build of the
# demo app, copied to demo/ when JBA_DEMO_WEB names their folder. Without it the
# demo page says the demo is not in this build. A relative path is from the repository root.
DEMO_WEB = os.environ.get("JBA_DEMO_WEB", "")
if DEMO_WEB:
    DEMO_WEB = os.path.join(os.path.dirname(HERE), DEMO_WEB)
DEMO_FILES = ("index.html", "jba_demo.js", "jba_demo.wasm")
if DEMO_WEB:
    missing = [f for f in DEMO_FILES if not os.path.isfile(os.path.join(DEMO_WEB, f))]
    if missing:
        raise RuntimeError(f"JBA_DEMO_WEB={DEMO_WEB} lacks {', '.join(missing)}")
    tags.add("demo")  # noqa: F821  (Sphinx provides `tags` to conf.py)


def copy_demo(app, exception):
    if exception is None and DEMO_WEB and app.builder.format == "html":
        import shutil

        out = os.path.join(app.outdir, "demo")
        os.makedirs(out, exist_ok=True)
        for f in DEMO_FILES:
            shutil.copy2(os.path.join(DEMO_WEB, f), out)


def setup(app):
    app.connect("build-finished", copy_demo)


# The site's own theme in _theme/justba (D130), on Sphinx's basic theme.
html_theme = "justba"
html_theme_path = ["_theme"]
html_title = f"just-ba {release}"
