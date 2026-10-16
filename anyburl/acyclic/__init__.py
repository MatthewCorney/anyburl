"""Acyclic rules of length one, end to end.

AnyBURL's AC1 and AC2 rules with a single body atom, ``r(X, c) :- b(X, d)``
and ``r(X, c) :- b(X, A)``, need none of the body-chain machinery: they are
sampled as one-step walks (:mod:`.sampling`), generalized from those steps
(:mod:`.generalization`), evaluated by direct adjacency lookups
(:mod:`.evaluation`) and fired individually at prediction time
(:mod:`.scoring`).
"""

from .evaluation import AcyclicIndex, evaluate_acyclic
from .generalization import AcyclicStep, WalkAnchor, generalize_acyclic
from .sampling import AcyclicStepSampler
from .scoring import AcyclicScorer

__all__ = [
    "AcyclicIndex",
    "AcyclicScorer",
    "AcyclicStep",
    "AcyclicStepSampler",
    "WalkAnchor",
    "evaluate_acyclic",
    "generalize_acyclic",
]
