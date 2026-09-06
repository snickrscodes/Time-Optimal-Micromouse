CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -ffp-contract=fast -fno-fast-math -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections
CPPFLAGS ?= -I. -Iinclude -I../segment/include
LDFLAGS ?= -shared -Wl,--gc-sections -Wl,-rpath,'$$ORIGIN/../segment:$$ORIGIN/../cflow:$$ORIGIN/../reverse_eta'
LDLIBS ?= -L../segment -lame_segment -lm
TARGET = libame_crossing.so
OBJS = crossing.o
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)
clean:
	rm -f $(OBJS) $(TARGET)
.PHONY: all clean benchmark
benchmark: $(TARGET)
	$(CXX) -O3 -std=c++17 -I. -Iinclude -I../segment/include benchmark.cpp -L. -lame_crossing -L../segment -lame_segment -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../segment:$$ORIGIN/../cflow:$$ORIGIN/../reverse_eta' -lm -o /tmp/ame_crossing_v1_benchmark
	LD_LIBRARY_PATH=$(CURDIR):$(CURDIR)/../segment:$(CURDIR)/../cflow:$(CURDIR)/../reverse_eta:$$LD_LIBRARY_PATH /tmp/ame_crossing_v1_benchmark
