"""Route optimization models (slides Step 3).

Main testing models:
  1. MIPSSTW + MCS
  2. Composite DRL
  3. GBDT (LightGBM) edge-cost router
"""

from .conditions import (
    CONDITION_PRESETS,
    RoutingConditions,
    WEATHER_PROFILES,
    conditions_at,
    congestion_for_hour,
    get_conditions,
    list_condition_ids,
    load_open_meteo_snapshot,
    pick_harsh_open_meteo_hours,
)
from .emv_corridors import annotate_emv_road_privileges, path_corridor_segments
from .mipsstw_mcs import solve_mipsstw_mcs
from .composite_drl import solve_composite_drl
from .gbdt_router import solve_gbdt_route, train_gbdt_edge_model, build_edge_training_frame
from .graph import (
    RouteResult,
    apply_routing_conditions,
    control_civilian_time,
    control_google_maps,
    control_shortest_distance,
    nearest_node,
    prepare_routing_graph,
)

__all__ = [
    "RouteResult",
    "prepare_routing_graph",
    "apply_routing_conditions",
    "nearest_node",
    "control_shortest_distance",
    "control_civilian_time",
    "control_google_maps",
    "solve_mipsstw_mcs",
    "solve_composite_drl",
    "solve_gbdt_route",
    "train_gbdt_edge_model",
    "build_edge_training_frame",
    "RoutingConditions",
    "CONDITION_PRESETS",
    "WEATHER_PROFILES",
    "get_conditions",
    "conditions_at",
    "congestion_for_hour",
    "list_condition_ids",
    "load_open_meteo_snapshot",
    "pick_harsh_open_meteo_hours",
    "annotate_emv_road_privileges",
    "path_corridor_segments",
]
