#ifndef AME_REVERSE_H
#define AME_REVERSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
# define AME_REVERSE_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
# define AME_REVERSE_API __attribute__((visibility("default")))
#else
# define AME_REVERSE_API
#endif

typedef enum {
    AME_REVERSE_OK = 0,
    AME_REVERSE_INVALID_ARGUMENT = 1,
    AME_REVERSE_DOMAIN = 2,
    AME_REVERSE_CONDITIONING = 3,
    AME_REVERSE_NUMERICAL_FAILURE = 4,
    AME_REVERSE_TOPOLOGY_FAILURE = 5,
    AME_REVERSE_ALLOCATION_FAILURE = 6,
    AME_REVERSE_UNSUPPORTED = 7
} ame_reverse_status;

typedef enum {
    AME_REVERSE_PASS_FORWARD = 1,
    AME_REVERSE_PASS_BACKWARD = 2
} ame_reverse_pass_kind;

typedef enum {
    AME_REVERSE_EVENT_PIECE_END = 1,
    AME_REVERSE_EVENT_GRIP_MOTOR = 2,
    AME_REVERSE_EVENT_MOTOR_GRIP = 3,
    AME_REVERSE_EVENT_GRIP_BRAKE = 4,
    AME_REVERSE_EVENT_BRAKE_GRIP = 5
} ame_reverse_event_kind;

typedef enum {
    AME_REVERSE_MODE_GRIP = 1,
    AME_REVERSE_MODE_MOTOR = 2,
    AME_REVERSE_MODE_BRAKE = 3
} ame_reverse_mode;

typedef struct ame_reverse_options {
    int has_init_w;
    double init_w;
    int has_terminal_w_max;
    double terminal_w_max;
    double initial_k;
    int has_backward_init_k;
    double backward_init_k;
    int n_scan;
    int domain_scan;
    double domain_margin;
    int fused_grip_discovery;
    int validate_replay_domain;
} ame_reverse_options;

typedef struct ame_scalar_segment_view {
    int pass_index;
    ame_reverse_pass_kind pass_kind;
    ame_reverse_mode mode;
    ame_reverse_event_kind event;
    int traversal_index;
    int piece_index;
    int initial_knot_index;
    int boundary_start;
    double offset0;
    double L_used;
    double sigma;
    double abs0;
    double abs1;
    double direction;
    double w0;
    double k0;
    double w1;
    double k1;
} ame_scalar_segment_view;

typedef struct ame_envelope_piece_view {
    ame_reverse_pass_kind source;
    int source_index;
    int pass_index;
    double abs0;
    double abs1;
    double local0;
    double local1;
} ame_envelope_piece_view;

typedef struct ame_scalar_build_stats {
    size_t raw_values;
    size_t pieces;
    size_t scalar_passes;
    size_t scalar_segments;
    size_t envelope_pieces;
    size_t possible_anchors;
    size_t inserted_anchors;
    size_t anchor_rounds;
    uint64_t segment_compiles;
    uint64_t crossing_calls;
    uint64_t build_cflow_calls;
    uint64_t build_cflow_local_steps;
    uint64_t scalar_cflow_calls;
    uint64_t scalar_cflow_local_steps;
} ame_scalar_build_stats;

typedef struct ame_reverse_stats {
    size_t promoted_passes;
    size_t promoted_segments;
    size_t promoted_grip_segments;
    uint64_t replay_cflow_calls;
    uint64_t replay_cflow_local_steps;
} ame_reverse_stats;

typedef struct ame_scalar_build ame_scalar_build;
typedef struct ame_reverse_build ame_reverse_build;

AME_REVERSE_API ame_reverse_options ame_reverse_default_options(void);

/* Scalar/topology layer. No differentiable segment is constructed here. */
AME_REVERSE_API ame_reverse_status ame_scalar_build_create(
    const double *raw_params,
    size_t raw_count,
    const ame_reverse_options *options,
    ame_scalar_build **out_build
);
AME_REVERSE_API void ame_scalar_build_destroy(ame_scalar_build *build);
AME_REVERSE_API ame_reverse_status ame_scalar_build_get_stats(const ame_scalar_build *build, ame_scalar_build_stats *out);
AME_REVERSE_API size_t ame_scalar_build_pass_count(const ame_scalar_build *build);
AME_REVERSE_API size_t ame_scalar_build_segment_count(const ame_scalar_build *build);
AME_REVERSE_API size_t ame_scalar_build_envelope_count(const ame_scalar_build *build);
AME_REVERSE_API size_t ame_scalar_build_inserted_anchor_count(const ame_scalar_build *build);
AME_REVERSE_API int ame_scalar_build_inserted_anchor_at(const ame_scalar_build *build, size_t index);
AME_REVERSE_API ame_reverse_status ame_scalar_build_segment_at(const ame_scalar_build *build, size_t flat_index, ame_scalar_segment_view *out);
AME_REVERSE_API ame_reverse_status ame_scalar_build_envelope_at(const ame_scalar_build *build, size_t index, ame_envelope_piece_view *out);
AME_REVERSE_API ame_reverse_status ame_scalar_build_time_value(ame_scalar_build *build, double *out_value);
AME_REVERSE_API ame_reverse_status ame_scalar_build_time_value_gradient(ame_scalar_build *build, double *out_value, double *out_grad, size_t grad_count);

/* Differentiable promotion layer. Scalar build remains authoritative.
 * The scalar build must outlive every ame_reverse_build promoted from it. */
AME_REVERSE_API ame_reverse_status ame_reverse_promote_time(
    ame_scalar_build *scalar,
    ame_reverse_build **out_reverse
);
AME_REVERSE_API ame_reverse_status ame_reverse_promote_full(
    ame_scalar_build *scalar,
    ame_reverse_build **out_reverse
);
AME_REVERSE_API void ame_reverse_build_destroy(ame_reverse_build *build);
AME_REVERSE_API ame_reverse_status ame_reverse_build_get_stats(const ame_reverse_build *build, ame_reverse_stats *out);

/* Reverse objectives. Output gradient length must equal raw_count. */
AME_REVERSE_API ame_reverse_status ame_reverse_time_value_gradient(
    ame_reverse_build *build,
    double *out_value,
    double *out_gradient,
    size_t gradient_count
);
AME_REVERSE_API ame_reverse_status ame_reverse_final_state_rows(
    ame_reverse_build *build,
    ame_reverse_pass_kind which,
    double *out_w_row,
    double *out_k_row,
    size_t row_count
);

/* One-shot convenience; still uses separate scalar build and promotion internally. */
AME_REVERSE_API ame_reverse_status ame_reverse_time_value_gradient_raw(
    const double *raw_params,
    size_t raw_count,
    const ame_reverse_options *options,
    double *out_value,
    double *out_gradient,
    size_t gradient_count
);

AME_REVERSE_API const char *ame_reverse_last_error(void);
AME_REVERSE_API const char *ame_reverse_status_name(ame_reverse_status status);

#ifdef __cplusplus
}
#endif
#endif
