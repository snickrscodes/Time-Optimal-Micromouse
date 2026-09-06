CC ?= cc
CFLAGS ?= -O3 -std=c11 -fPIC -Wall -Wextra -Wpedantic -fno-fast-math
LDFLAGS ?= -shared
all: libame_dd_yaw.so
libame_dd_yaw.so: dd_yaw.c ame_dd_yaw.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ dd_yaw.c -lm
clean:
	rm -f libame_dd_yaw.so
audit: all
	ldd ./libame_dd_yaw.so || true
	nm -D --defined-only ./libame_dd_yaw.so | grep ' ame_dd_' || true
.PHONY: all clean audit
