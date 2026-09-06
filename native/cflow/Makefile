CC ?= cc
# Production Cflow requires an optimized native kernel.  Ambient/command-line
# CFLAGS are still accepted for architecture, warning, and packaging flags, but
# they must not silently erase the production optimization baseline.  Use
# CFLOW_OPT_CFLAGS=-O0 explicitly for a debug build.
CFLOW_OPT_CFLAGS ?= -O3
CFLOW_REQUIRED_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=fast -fno-fast-math
override CFLAGS += $(CFLOW_OPT_CFLAGS) $(CFLOW_REQUIRED_CFLAGS)
override CPPFLAGS += -Iinclude -Isrc -Igenerated
LDFLAGS ?=
LIBS = -lm
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHARED = libcflow.dylib
  SHARED_FLAGS = -dynamiclib
else
  SHARED = libcflow.so
  SHARED_FLAGS = -shared -fPIC
  override CFLAGS += -fPIC
  override LDFLAGS += -Wl,--gc-sections
endif
override CFLAGS += -fvisibility=hidden -ffunction-sections -fdata-sections
SRC = src/cflow.c src/cflow_core.c src/cflow_highz.c src/cflow_terminal.c \
      src/cflow_farfield.c src/cflow_event.c src/cflow_exact.c src/cflow_rk.c \
      src/cflow_tensor.c src/cflow_integral.c src/cflow_integral_terminal.c \
      src/cflow_boundary.c generated/core_coeffs.c generated/highz_coeffs.c \
      generated/highz_tucker_coeffs.c generated/highz_face_tables.c \
      generated/farfield_coeffs.c generated/integral_tables.c generated/quadrature.c
OBJ = $(SRC:.c=.o)
.PHONY: all clean audit print-config
all: $(SHARED)
$(SHARED): $(OBJ)
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LIBS)
audit: $(SHARED)
	@command -v ldd >/dev/null 2>&1 && ldd ./$(SHARED) || true
	@command -v nm >/dev/null 2>&1 && nm -D --defined-only ./$(SHARED) | grep ' cflow_' || true
clean:
	rm -f $(OBJ) $(SHARED)

print-config:
	@echo "CC=$(CC)"
	@echo "CFLOW_OPT_CFLAGS=$(CFLOW_OPT_CFLAGS)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"
