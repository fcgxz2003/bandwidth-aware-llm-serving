"""Path bootstrap so the flat-module scripts work from any subfolder.

The experiment code uses flat imports (``import config``, ``import expcommon``).
After the reorg into ``common/``, ``exp/`` and ``plot/``, a script started as
``python exp/exp_offline.py`` only has its own folder on ``sys.path``. Importing
this module (after putting the experiment root on the path) registers the
experiment root and the ``common``/``exp``/``plot`` folders, so every flat
import resolves regardless of where the script lives.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (
    _ROOT,
    os.path.join(_ROOT, "common"),
    os.path.join(_ROOT, "exp"),
    os.path.join(_ROOT, "plot"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
