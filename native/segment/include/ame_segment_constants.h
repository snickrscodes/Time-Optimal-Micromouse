#ifndef AME_SEGMENT_CONSTANTS_H
#define AME_SEGMENT_CONSTANTS_H

/*
 * Physical constants shared by the native Segment and Crossing research layers.
 * Keep derived switching thresholds out of this header: callers must recompute
 * them from these base constants so changed vehicle constants cannot leave stale
 * geometry literals behind.
 */
#define AME_SEGMENT_MU_G (1.2 * 9.81)
#define AME_SEGMENT_A_BRAKE (0.9 * AME_SEGMENT_MU_G)
#define AME_SEGMENT_A_MAX 15.0
#define AME_SEGMENT_V_MAX 4.0
#define AME_SEGMENT_B_EMF (AME_SEGMENT_A_MAX / AME_SEGMENT_V_MAX)

#endif
