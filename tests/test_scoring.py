"""Tests for grounding weighting and its former values."""

import pytest

from anyburl.prediction import GroundingWeighting


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("once", GroundingWeighting.ONCE),
        ("log_count", GroundingWeighting.LOG_COUNT),
        ("noisy_or", GroundingWeighting.ONCE),
        ("path_weighted", GroundingWeighting.LOG_COUNT),
    ],
)
def test_parses_current_and_former_values(
    text: str, expected: GroundingWeighting
) -> None:
    assert GroundingWeighting(text) is expected


def test_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="not a valid GroundingWeighting"):
        GroundingWeighting("maximum")
