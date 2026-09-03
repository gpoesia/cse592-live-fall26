from dataclasses import dataclass


@dataclass
class Node:
    state: object
    # Node we used to reach this node.
    parent: "Node" = None
    # Action from parent to this state.
    action: object = None
    # Node's cost.
    cost: float = 0.0

    def path(self) -> list['action']:
        """Return the actions from the initial state to this node."""
        actions, node = [], self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions


class SearchAlgorithm:
    def __init__(self):
        self.expanded = 0

    def search(self, env) -> list['action']:
        raise NotImplementedError
