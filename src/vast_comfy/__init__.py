"""vast-comfy: a Python toolkit for renting and preparing a Vast.ai ComfyUI instance.

See SPEC.md for the approved v1 scope. This package implements the control-side
toolkit; the small management components that run on the rented instance live
under ``vast_comfy/remote_agent`` and are deployed through the whitelisted
packaging in ``vast_comfy.remote``.
"""

from __future__ import annotations

__version__ = "0.1.0"

TOOL_NAME = "vast-comfy"
