from dataclasses import dataclass


@dataclass
class Transition:
    state: object
    action: object
    next_state: object
    reward: float


class Environment:
    """Interface exposed to a search algorithm."""

    def __init__(self, log=False):
        self.log = [] if log else None

    def initial_state(self):
        raise NotImplementedError

    def actions(self, state) -> list[tuple[object, int]]:
        """Return the available (action, cost) pairs."""
        raise NotImplementedError

    def is_goal(self, state) -> bool:
        raise NotImplementedError

    def transition(self, state, action) -> tuple[object, bool]:
        next_state, reward = self._transition(state, action)
        if self.log is not None:
            self.log.append(Transition(state, action, next_state, reward))
        return next_state, reward

    def _transition(self, state, action) -> tuple[object, bool]:
        raise NotImplementedError
