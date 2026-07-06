NVCC      ?= nvcc
ARCH      ?= -gencode arch=compute_86,code=sm_86
NVCCFLAGS  = -O3 -std=c++17 $(ARCH) -Xcompiler -O3 -lineinfo
BIN        = bin

TARGETS = $(BIN)/device_info $(BIN)/gemv $(BIN)/softmax $(BIN)/rmsnorm \
          $(BIN)/silu $(BIN)/attention

all: $(TARGETS)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/device_info: src/device_info.cu src/common.cuh | $(BIN)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN)/%: src/run_%.cu src/kernels/%.cuh src/common.cuh | $(BIN)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Run every op: correctness (vs fp64 CPU reference) + benchmarks.
run: all
	./$(BIN)/device_info
	@echo
	./$(BIN)/gemv 0
	@echo
	./$(BIN)/softmax 0
	@echo
	./$(BIN)/rmsnorm 0
	@echo
	./$(BIN)/silu 0
	@echo
	./$(BIN)/attention 0

clean:
	rm -rf $(BIN)

.PHONY: all run clean
