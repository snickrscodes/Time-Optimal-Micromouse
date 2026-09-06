CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -ffp-contract=fast -fno-fast-math -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections
CPPFLAGS ?= -I. -I../segment/include -I../crossing/include
LDFLAGS ?= -shared -Wl,--gc-sections -Wl,-rpath,'$$ORIGIN/../crossing:$$ORIGIN/../segment:$$ORIGIN/../cflow:$$ORIGIN/../reverse_eta'
LDLIBS ?= -L../crossing -lame_crossing -L../segment -lame_segment -lm
TARGET = libame_reverse.so
OBJS = reverse.o
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)
clean:
	rm -f $(OBJS) $(TARGET)
audit: $(TARGET)
	ldd ./$(TARGET) || true
	nm -D --defined-only ./$(TARGET) | grep ' ame_\(scalar\|reverse\)_' || true
.PHONY: all clean audit
