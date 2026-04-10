"""Tests for communication action index mapping."""

import pytest

from policy.actions import NO_COMMUNICATION, partner_to_selection, selection_to_partner


def test_selection_maps_to_other_agents_in_sorted_order() -> None:
    assert selection_to_partner(NO_COMMUNICATION, agent_id=1, num_agents=4) is None
    assert selection_to_partner(selection=1, agent_id=1, num_agents=4) == 0
    assert selection_to_partner(selection=2, agent_id=1, num_agents=4) == 2
    assert selection_to_partner(selection=3, agent_id=1, num_agents=4) == 3


def test_partner_maps_back_to_local_selection() -> None:
    assert partner_to_selection(partner_id=0, agent_id=1, num_agents=4) == 1
    assert partner_to_selection(partner_id=2, agent_id=1, num_agents=4) == 2
    assert partner_to_selection(partner_id=3, agent_id=1, num_agents=4) == 3


def test_self_partner_is_invalid() -> None:
    with pytest.raises(ValueError, match="cannot select itself"):
        partner_to_selection(partner_id=1, agent_id=1, num_agents=4)


def test_selection_must_be_valid_action_index() -> None:
    with pytest.raises(ValueError, match=r"\[0, num_agents\)"):
        selection_to_partner(selection=4, agent_id=1, num_agents=4)
