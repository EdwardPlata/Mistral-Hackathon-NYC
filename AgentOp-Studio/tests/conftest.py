"""Pytest configuration — ensures AgentOp-Studio packages are importable.

Because the directory name contains a hyphen (not valid in Python identifiers),
we add the AgentOp-Studio directory itself to sys.path so that tests can do
``import backend.db`` or ``import agents.tools`` directly.
"""

import os
import sys

# AgentOp-Studio directory — gives access to both `backend.*` and `agents.*`
_studio_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _studio_dir not in sys.path:
    sys.path.insert(0, _studio_dir)
