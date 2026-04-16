"""Tests for binary communication actions."""

import pytest

from policy.actions import (
    BINARY_ACTION_SIZE,
    COMMUNICATE,
    NO_COMMUNICATION,
    is_communication,
)


def test_binary_action_constants() -> None:
    assert NO_COMMUNICATION == 0
    assert COMMUNICATE == 1
    assert BINARY_ACTION_SIZE == 2


def test_is_communication_identifies_positive_selection() -> None:
    assert not is_communication(NO_COMMUNICATION)
    assert is_communication(COMMUNICATE)


def test_is_communication_requires_binary_selection() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        is_communication(2)
