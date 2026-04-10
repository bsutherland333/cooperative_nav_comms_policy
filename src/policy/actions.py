"""Index mapping for discrete communication actions."""

NO_COMMUNICATION = 0


def selection_to_partner(selection: int, agent_id: int, num_agents: int) -> int | None:
    """Map a local action index to the selected global partner ID."""
    _validate_agent_id(agent_id, num_agents)
    _validate_selection(selection, num_agents)
    if selection == NO_COMMUNICATION:
        return None

    partner_id = selection - 1
    if partner_id >= agent_id:
        partner_id += 1
    return partner_id


def partner_to_selection(partner_id: int, agent_id: int, num_agents: int) -> int:
    """Map a global partner ID to the acting agent's local action index."""
    _validate_agent_id(agent_id, num_agents)
    _validate_agent_id(partner_id, num_agents)
    if partner_id == agent_id:
        raise ValueError("An agent cannot select itself as a communication partner.")

    if partner_id < agent_id:
        return partner_id + 1
    return partner_id


def _validate_agent_id(agent_id: int, num_agents: int) -> None:
    if num_agents < 2:
        raise ValueError("At least two agents are required.")
    if agent_id < 0 or agent_id >= num_agents:
        raise ValueError("Agent ID is outside the fleet.")


def _validate_selection(selection: int, num_agents: int) -> None:
    if selection < 0 or selection >= num_agents:
        raise ValueError("Action selection must be in [0, num_agents).")
