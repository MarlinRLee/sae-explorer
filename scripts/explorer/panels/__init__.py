"""Self-contained Bokeh panels.

Each module in this package exports a ``build()`` factory that constructs
fresh widgets (per Bokeh session), wires their internal callbacks, and
returns the panel layout plus any widgets the bootstrap needs to access
from elsewhere.

The bootstrap orchestrates: it constructs ``state`` / ``ui`` / ``args``,
publishes them to :mod:`explorer.runtime`, calls each panel's
``build()``, and assembles the final layout.
"""
