CC ?= cc
REVERSE_ETA_OPT_CFLAGS ?= -O3
REVERSE_ETA_REQUIRED_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=fast -fno-fast-math
override CFLAGS += $(REVERSE_ETA_OPT_CFLAGS) $(REVERSE_ETA_REQUIRED_CFLAGS)
override CPPFLAGS += -I.
LDFLAGS ?=
LIBS = -lm
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHARED = libreverse_eta.dylib
  SHARED_FLAGS = -dynamiclib
else
  SHARED = libreverse_eta.so
  SHARED_FLAGS = -shared -fPIC
  override CFLAGS += -fPIC
  override LDFLAGS += -Wl,--gc-sections
endif
override CFLAGS += -fvisibility=hidden -ffunction-sections -fdata-sections
SRC = reverse_eta.c
OBJ = $(SRC:.c=.o)
.PHONY: all clean audit print-config
all: $(SHARED)
$(SHARED): $(OBJ)
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LIBS)
audit: $(SHARED)
	@command -v ldd >/dev/null 2>&1 && ldd ./$(SHARED) || true
	@command -v nm >/dev/null 2>&1 && nm -D --defined-only ./$(SHARED) | grep ' ame_reverse_eta_' || true
clean:
	rm -f $(OBJ) $(SHARED)
print-config:
	@echo "CC=$(CC)"
	@echo "REVERSE_ETA_OPT_CFLAGS=$(REVERSE_ETA_OPT_CFLAGS)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"
