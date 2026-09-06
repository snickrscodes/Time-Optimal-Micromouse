"""Low-allocation undirected graph block decomposition utilities."""

from __future__ import annotations

from mazegen import JunctionGraph

EdgeKey = tuple[int, int]


def undirected_edges(graph: JunctionGraph) -> tuple[EdgeKey, ...]:
    """Return each undirected junction edge once in canonical order."""
    return tuple(
        (source, edge.to)
        for source, edges in enumerate(graph.adj)
        for edge in edges
        if source < edge.to
    )


def biconnected_edge_components(graph: JunctionGraph) -> list[list[EdgeKey]]:
    """Iterative Tarjan decomposition into edge-biconnected blocks.

    Bridges are returned as one-edge blocks.  The implementation avoids Python
    recursion so its stack use is independent of maze size.
    """
    n = len(graph.nodes)
    discovery = [-1] * n
    low = [0] * n
    parent = [-1] * n
    clock = 0
    edge_stack: list[EdgeKey] = []
    components: list[list[EdgeKey]] = []

    for root in range(n):
        if discovery[root] >= 0:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            node, next_index = stack[-1]
            if next_index < len(graph.adj[node]):
                edge = graph.adj[node][next_index]
                stack[-1] = (node, next_index + 1)
                neighbor = edge.to
                if discovery[neighbor] < 0:
                    parent[neighbor] = node
                    discovery[neighbor] = low[neighbor] = clock
                    clock += 1
                    edge_stack.append((node, neighbor))
                    stack.append((neighbor, 0))
                elif neighbor != parent[node] and discovery[neighbor] < discovery[node]:
                    low[node] = min(low[node], discovery[neighbor])
                    edge_stack.append((node, neighbor))
                continue

            stack.pop()
            ancestor = parent[node]
            if ancestor < 0:
                if edge_stack:
                    components.append(edge_stack[:])
                    edge_stack.clear()
                continue
            low[ancestor] = min(low[ancestor], low[node])
            if low[node] >= discovery[ancestor]:
                component: list[EdgeKey] = []
                while edge_stack:
                    item = edge_stack.pop()
                    component.append(item)
                    if item == (ancestor, node):
                        break
                if component:
                    components.append(component)
    return components


__all__ = ["EdgeKey", "biconnected_edge_components", "undirected_edges"]
