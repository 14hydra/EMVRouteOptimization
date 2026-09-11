"""Route optimization models (slides Step 3).

Main testing models:
  1. MIPSSTW + MCS
  2. Composite DRL
  3. GBDT (LightGBM) edge-cost router
"""

from .graph import (
    RouteResult,
    control_civilian_time,
    control_shortest_distance,
    nearest_node,
    prepare_routing_graph,
)
from .conditions import (
    CONDITION_PRESETS,
    RoutingConditions,
    get_conditions,
    list_condition_ids,
    load_open_meteo_snapshot,
    pick_harsh_open_meteo_hours,
)
from .mipsstw_mcs import solve_mipsstw_mcs
from .composite_drl import solve_composite_drl
from .gbdt_router import solve_gbdt_route, train_gbdt_edge_model, build_edge_training_frame

__all__ = [
    "RouteResult",
    "prepare_routing_graph",
    "nearest_node",
    "control_shortest_distance",
    "control_civilian_time",
    "solve_mipsstw_mcs",
    "solve_composite_drl",
    "solve_gbdt_route",
    "train_gbdt_edge_model",
    "build_edge_training_frame",
    "RoutingConditions",
    "CONDITION_PRESETS",
    "get_conditions",
    "list_condition_ids",
    "load_open_meteo_snapshot",
    "pick_harsh_open_meteo_hours",
]
