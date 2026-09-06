from .certification import certify_parameters, certify_parameters_on_problem

from .sparse_route_activation import (
    SparseRouteEligibilityDecision,
    SparseRouteEligibilitySettings,
    SparseRouteFeatures,
    count_route_turns,
    evaluate_sparse_route_eligibility,
)
"""Certified topology search and continuous-route construction."""

from .branch_and_bound import (
    BranchAndBoundResult,
    CompleteCandidateObserver,
    CompletePathEvaluation,
    SearchNode,
    SearchSettings,
    SearchTraceEvent,
    SearchTraceObserver,
    branch_and_bound_junction_paths,
)
from .maze_routes import (
    DEFAULT_BODY_HEIGHT,
    DEFAULT_BODY_LENGTH,
    OpenRoomSpan,
    RouteOptimizationProblem,
    TURN_HALF_LENGTH,
    TURN_PEAK_CURVATURE,
    astar_junction_path,
    build_route_optimization_problem,
    expand_junction_edge,
    expand_junction_path,
    legacy_stable_knot_bounds,
    optimize_route_curvature,
    shortest_cell_path,
    shortest_junction_path,
    stable_knot_bounds,
)
from .time_bounds import (
    AxisAlignedBox,
    OrderedBoxDistanceResult,
    OrderedPortalDistanceResult,
    Portal,
    PortalMotorTimeLowerBound,
    CompleteTimeLowerBound,
    TimeBoundRequest,
    TimeBoundResult,
    TimeBoundStatistics,
    TimeLowerBound,
    clearance_gate,
    eroded_cover_distance_bounds,
    mandatory_projection_extrema,
    motor_only_time,
    multidirectional_projection_time_lower_bound,
    ordered_box_distance_bounds,
    ordered_portal_distance_bounds,
    portal_between_cells,
    projected_reversal_time_lower_bound,
    portals_from_cell_path,
    straight_two_sided_time_lower_bound,
)

from .topology_quotient import (
    OpenRoomBlock,
    OpenRoomTopologyQuotient,
    QuotientRouteClass,
    QuotientRouteVariant,
    TopologyClassKey,
    TopologyQuotientStatistics,
)

from .warm_start_schedule import (
    BasinComparison,
    PrimaryFilterSQPPolicySettings,
    PrimaryTimeBackendMode,
    TimePolishCheckpoint,
    TimePolishNumericalCheckpoint,
    TimePolishCheckpointSettings,
    WarmStartDeadlineSettings,
    WarmStartExecutionResult,
    WarmStartInitializerMode,
    WarmStartOptimizationConfig,
    WarmStartRunRecord,
    WarmStartScheduleSettings,
    WarmStartSchedulingMode,
    WarmStartStageRecord,
    WarmStartStageRequest,
    WarmStartStageResult,
    compare_basin_candidates,
    create_warm_start_stage_runner,
    execute_bounded_warm_start_schedule,
    run_supervised_warm_start_stage,
)

from .route_optimization_policy import (
    ActiveBasisPlannerSettings,
    PlannerArchitectureMode,
    PlannerOptimizationPolicy,
    SparseSpecialistMode,
    SparseSpecialistPolicySettings,
)

from .sparse_specialist import (
    SparseSpecialistExecutionResult,
    SparseSpecialistRecord,
    run_sparse_specialist_polish,
)

__all__ = [name for name in globals() if not name.startswith("_")]

from .maze_scenario import GoalEntrance, GoalRegion, MazeScale, MazeScenario
from .maze_io import FORMAT_ID as MAZE_FORMAT_ID, MazeFormatError, load_maze_scenario
