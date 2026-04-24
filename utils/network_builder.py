"""
utils/network_builder.py
------------------------
Constructs NetworkX multi-line graph for the 13 Regional Rail lines.
Used for:
  - Path enumeration (lower level logit)
  - Express vs local routing (upper level extension)
  - Visualization

GRAPH STRUCTURE:
  Nodes: all unique station names across all 13 lines
  Edges: directed, one per (station_i → station_j, line) triple
         bidirectional (inbound + outbound)
  Node attributes: type ∈ {terminal, through, transfer, center_city}
  Edge attributes: line, travel_time, type ∈ {line_arc}
"""

import networkx as nx
from collections import defaultdict
from data.network import LINES, CENTER_CITY_STATIONS


def build_network() -> tuple[nx.DiGraph, dict]:
    """
    Returns: (G, station_lines_map)
      G                : directed multigraph
      station_lines_map: {station_name: [line_names]}
    """
    G = nx.DiGraph()
    station_lines = defaultdict(list)

    for line_name, line_data in LINES.items():
        stations = line_data["stations"]
        times    = line_data["travel_times"]

        for i, s in enumerate(stations):
            station_lines[s].append(line_name)
            if s not in G:
                G.add_node(s, type="through", lines=[])
            G.nodes[s]["lines"] = list(set(G.nodes[s]["lines"] + [line_name]))

            if i < len(stations) - 1:
                tt = times[i] if i < len(times) else 4
                # Both directions
                for u, v in [(stations[i], stations[i+1]),
                             (stations[i+1], stations[i])]:
                    G.add_edge(u, v, line=line_name, travel_time=tt,
                               type="line_arc", capacity=875)

    # Tag node types
    for s in G.nodes:
        lines = station_lines[s]
        if s in CENTER_CITY_STATIONS:
            G.nodes[s]["type"] = "center_city"
        elif len(lines) > 1:
            G.nodes[s]["type"] = "transfer"
        elif any(s == LINES[l]["stations"][0] or s == LINES[l]["stations"][-1]
                 for l in lines):
            G.nodes[s]["type"] = "terminal"

    return G, dict(station_lines)


def shortest_path_travel_time(G: nx.DiGraph, origin: str, dest: str) -> float:
    """Returns total travel time (minutes) on shortest-time path."""
    try:
        path = nx.shortest_path(G, origin, dest, weight="travel_time")
        return sum(G[path[i]][path[i+1]]["travel_time"] for i in range(len(path)-1))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 999.0   # unreachable
