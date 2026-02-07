"""Enumerations for AnyBURL rule types, term kinds, and strategies."""

from enum import Enum, StrEnum, auto


class RuleType(StrEnum):
    """Classification of learned Horn rules by structural type.

    Attributes
    ----------
    AC1 : str
        Acyclic rule with one free variable. One head variable is grounded
        as a constant; the other appears in the body.
    AC2 : str
        Acyclic rule with two free variables. One head variable does not
        appear in the body at all.
    CYCLIC : str
        Cyclic rule. Both head variables appear in the body, forming a
        closed variable chain.
    """

    AC1 = "ac1"
    AC2 = "ac2"
    CYCLIC = "cyclic"


class TermKind(Enum):
    """Whether a term in an atom is a variable or a grounded constant.

    Attributes
    ----------
    VARIABLE : int
        A placeholder that can bind to any entity during rule application.
    CONSTANT : int
        A specific, fixed entity from the knowledge graph.
    """

    VARIABLE = auto()
    CONSTANT = auto()


class WalkStrategy(StrEnum):
    """Strategy for selecting the next edge during a random walk.

    Attributes
    ----------
    UNIFORM : str
        Select uniformly at random among all outgoing edges.
    RELATION_WEIGHTED : str
        Weight edges by inverse frequency of their relation type,
        favoring rarer relations to discover more diverse rules.
    """

    UNIFORM = "uniform"
    RELATION_WEIGHTED = "relation_weighted"


class SamplingStrategy(StrEnum):
    """Strategy for sampling target triples from the knowledge graph.

    Attributes
    ----------
    UNIFORM : str
        Each triple has equal probability of being sampled.
    RELATION_PROPORTIONAL : str
        Sample proportional to relation frequency. More common
        relations yield more samples.
    RELATION_INVERSE : str
        Sample inversely proportional to relation frequency. Rarer
        relations get more samples, improving coverage.
    """

    UNIFORM = "uniform"
    RELATION_PROPORTIONAL = "relation_proportional"
    RELATION_INVERSE = "relation_inverse"
