"""Constants for binary pairwise communication actions."""

NO_COMMUNICATION = 0
COMMUNICATE = 1
BINARY_ACTION_SIZE = 2


def is_communication(selection: int) -> bool:
    """Return whether a binary actor selection requests communication."""
    _validate_binary_selection(selection)
    return selection == COMMUNICATE


def _validate_binary_selection(selection: int) -> None:
    if selection not in (NO_COMMUNICATION, COMMUNICATE):
        raise ValueError("Binary communication selection must be 0 or 1.")
