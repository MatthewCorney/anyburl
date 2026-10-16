"""Hold out validation edges as well as test edges.

:func:`split_with_validation` runs :func:`~anyburl.split.split_target_edges`
twice: once to hold out the test triples, then again on the resulting
training graph to hold out validation triples. The test triples and the
training graph are identical to a plain two-way split with the same config,
so test metrics stay comparable with it. Mirrored inverse edges are removed
at both stages.
"""

from dataclasses import dataclass, replace

from torch_geometric.data import HeteroData

from ._logging import get_logger
from .core import Triple
from .exceptions import ConfigurationError
from .split import SplitConfig, split_target_edges

logger = get_logger(__name__)

DEFAULT_VALIDATION_FRACTION: float = 0.1
"""Share of the post-test training edges held out when none is given."""

VALIDATION_SEED_OFFSET: int = 1
"""Added to the split seed so validation edges are drawn independently."""


@dataclass(frozen=True, slots=True)
class ValidationSplit:
    """A training graph with both validation and test triples withheld.

    Parameters
    ----------
    fit_data : HeteroData
        The graph without validation or test edges. Mine rules on this.
    validation_triples : tuple[Triple, ...]
        Edges removed from ``fit_data`` only, for tuning or training a
        scorer on rules mined from ``fit_data``.
    train_data : HeteroData
        The graph without test edges, identical to the two-way split's
        training graph. Use it as the known-triple filter when ranking test
        triples, so validation edges count as known facts.
    test_triples : tuple[Triple, ...]
        Edges removed from both graphs, identical to the two-way split.
    """

    fit_data: HeteroData
    validation_triples: tuple[Triple, ...]
    train_data: HeteroData
    test_triples: tuple[Triple, ...]


def split_with_validation(
    data: HeteroData,
    config: SplitConfig,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> ValidationSplit:
    """Hold out test edges, then validation edges from what remains.

    Parameters
    ----------
    data : HeteroData
        The full graph. Not modified.
    config : SplitConfig
        The test split. Its target, inverse handling and seed also govern
        the validation split.
    validation_fraction : float
        Share of the post-test training edges to hold out, in ``(0.0, 1.0)``.

    Returns
    -------
    ValidationSplit
        The fitting graph, training graph, validation and test triples.

    Raises
    ------
    ConfigurationError
        If ``validation_fraction`` is outside ``(0.0, 1.0)`` or holds out no
        training edges or all of them.
    GraphSchemaError
        As raised by :func:`~anyburl.split.split_target_edges`.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigurationError(
            f"validation_fraction must be in (0.0, 1.0), got {validation_fraction}"
        )

    test_split = split_target_edges(data, config)
    validation_config = replace(
        config,
        test_fraction=validation_fraction,
        seed=config.seed + VALIDATION_SEED_OFFSET,
    )
    try:
        validation_split = split_target_edges(test_split.train_data, validation_config)
    except ConfigurationError as error:
        raise ConfigurationError(
            f"validation_fraction {validation_fraction} leaves no validation "
            f"or no fitting edges: {error}"
        ) from error

    logger.info(
        "Validation split: %d validation triples, %d test triples",
        len(validation_split.test_triples),
        len(test_split.test_triples),
    )
    return ValidationSplit(
        fit_data=validation_split.train_data,
        validation_triples=validation_split.test_triples,
        train_data=test_split.train_data,
        test_triples=test_split.test_triples,
    )
