import pytest
from pydantic import ValidationError

from src.core.config import GenerationConfig, Settings


def test_context_expansion_is_disabled_by_default() -> None:
    settings = Settings()

    assert (
        settings.generation.enable_context_expansion
        is False
    )


def test_neighbor_context_character_budget_defaults_to_6000() -> None:
    config = GenerationConfig()

    assert config.max_neighbor_context_chars == 6000


def test_negative_neighbor_context_character_budget_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationConfig(
            max_neighbor_context_chars=-1,
        )