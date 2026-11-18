# Documentation site

Sphinx, built from this folder (D119), with the site's own theme in `_theme/justba`
(D130). The pages under `_generated/` are written on
every build by `gen/generate.py` from the residual definitions, the `just_ba`
module and the stored benchmark baselines; they are not committed.

```bash
uv venv --python 3.12 docs-site/.venv && uv pip install --python docs-site/.venv/bin/python -r docs-site/requirements.txt
```

With a build of the module (`-DJBA_BUILD_PYTHON=ON`, here `build/dev`):

```bash
PYTHONPATH=build/dev/bindings/python docs-site/.venv/bin/sphinx-build -W -b html docs-site docs-site/_build/html
```

```bash
PYTHONPATH=build/dev/bindings/python docs-site/.venv/bin/sphinx-build -W -b doctest docs-site docs-site/_build/doctest
```

With `JBA_DEMO_WEB` set to a WebAssembly build of the demo app (its `web/` folder,
see `demo/README.md`; a relative path is from the repository root), the live demo is copied to `demo/` and shown on its page;
without it the page says the demo is not in this build. The `docs` workflow takes
the demo from the release `demo-web`, which `demo/tools/publish_web.sh` uploads (D148).

The second runs every `>>>` example in the docstrings. CI's `docs` workflow (started
by hand) does both and, with `deploy=true`, publishes to Cloudflare Pages.
