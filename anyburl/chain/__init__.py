"""Chain grounding engine: count body-chain groundings without materialising.

Internal machinery used by :mod:`anyburl.metrics` and the walk engine; not part
of the public API.
"""

from .scanner import ChainScanner
from .tables import CsrDirection, CsrTables, build_csr_tables

__all__ = ["ChainScanner", "CsrDirection", "CsrTables", "build_csr_tables"]
