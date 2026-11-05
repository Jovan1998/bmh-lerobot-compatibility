"""Deprecated alias of :mod:`bmh.policy_client.server_client` (see :mod:`bmh.groot_client`)."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("bmh.policy_client.server_client")
