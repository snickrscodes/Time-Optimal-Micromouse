CC ?= cc
SEGMENT_OPT_CFLAGS ?= -O3
ROOT := ../..
CFLOW := ../cflow
ETA := ../reverse_eta
CFLAGS += $(SEGMENT_OPT_CFLAGS) -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=fast -fno-fast-math -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections
CPPFLAGS += -I. -Iinclude -I$(CFLOW)/include -I$(ETA)
LDFLAGS += -Wl,--gc-sections -Wl,-rpath,'$$ORIGIN/../cflow:$$ORIGIN/../reverse_eta'
LDLIBS += -L$(CFLOW) -lcflow -L$(ETA) -lreverse_eta -lm
SRC := segment.c straight.c circular.c motor.c grip.c
OBJ := $(SRC:.c=.o)
SHARED := libame_segment.so
.PHONY: all clean audit
all: $(SHARED)
$(SHARED): $(OBJ) $(CFLOW)/libcflow.so $(ETA)/libreverse_eta.so
	$(CC) -shared $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)
$(CFLOW)/libcflow.so:
	$(MAKE) -C $(CFLOW)
$(ETA)/libreverse_eta.so:
	$(MAKE) -C $(ETA)
audit: $(SHARED)
	ldd ./$(SHARED) || true
	nm -D --defined-only ./$(SHARED) | grep ' ame_segment_' || true
clean:
	rm -f $(OBJ) $(SHARED)
