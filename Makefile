# GNU make build for the CUDA manufacturing course (Linux; developed for ASU Sol).
#
#   make help                     list every target
#   make cpu test-cpu             Lab 01 CPU reference only (no CUDA toolkit needed)
#   make                          CPU reference plus every CUDA lab (needs nvcc)
#   make test-lab02               one lab's tests (inside a GPU allocation)
#   make SOLUTION=1 test-lab02    the same tests against the reference solution
#   make CUDA_ARCH=86 ...         another GPU (A100 = 80); objects rebuild automatically
#
# Your kernels live in src/labNN/; reference solutions in solutions/labNN/. The two builds use
# separate directories (build/ and build-solution/), so switching never mixes object files.

CXX       ?= g++
NVCC      ?= nvcc
PYTHON    ?= python3
SANITIZER ?= compute-sanitizer
NSYS      ?= nsys
NCU       ?= ncu
CUDA_ARCH ?= 80
SOLUTION  ?= 0
OPT       ?= -O3

ifeq ($(SOLUTION),1)
KERNEL_ROOT := solutions
BUILD       ?= build-solution
else
KERNEL_ROOT := src
BUILD       ?= build
endif

CXXFLAGS  ?= $(OPT) -std=c++17 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= $(OPT) -std=c++17 -lineinfo -arch=sm_$(CUDA_ARCH) -Isrc -Xcompiler -Wall,-Wextra

OBJ       := $(BUILD)/obj
CHECK     := bash scripts/check.sh
MEMCHECK  := $(SANITIZER) --tool memcheck --error-exitcode 3
RACECHECK := $(SANITIZER) --tool racecheck --error-exitcode 3
SYNCCHECK := $(SANITIZER) --tool synccheck --error-exitcode 3
STAMP     := $(shell date +%Y%m%d-%H%M%S)
RUN_DIR   ?= runs/$(STAMP)

CPU_BINS  := $(BUILD)/calibrate_cpu
CUDA_BINS := $(BUILD)/calibrate_cuda $(BUILD)/calibrate_channels $(BUILD)/window_stats \
             $(BUILD)/screen $(BUILD)/image_ops $(BUILD)/box_filter $(BUILD)/score_images \
             $(BUILD)/reduce_compare $(BUILD)/profile_inspect $(BUILD)/stream_batches $(BUILD)/inspect \
             $(BUILD)/release_bench

COMMON_H := $(wildcard src/common/*.hpp src/common/*.cuh)
LAB02_H  := src/lab02/lab02.hpp $(COMMON_H)
LAB03_H  := src/lab03/lab03.hpp $(COMMON_H)
LAB04_H  := src/lab04/lab04.hpp $(LAB02_H) $(LAB03_H)
LAB05_H  := src/lab05/lab05.hpp $(COMMON_H)
LAB06_H  := src/lab06/lab06.hpp $(COMMON_H)
LAB07_H  := src/lab07/lab07.hpp $(LAB05_H) $(LAB06_H)
LAB08_H  := src/lab08/lab08.hpp $(COMMON_H)
LAB09_H  := src/lab09/lab09.hpp $(LAB07_H)
LAB10_H  := src/lab10/lab10.hpp $(LAB02_H) $(LAB03_H)
LAB11_H  := src/lab11/lab11.hpp $(LAB04_H) $(LAB09_H)
LAB12_H  := src/lab12/lab12.hpp $(LAB09_H)

.DEFAULT_GOAL := all
.PHONY: all cpu cuda help clean FORCE lab01 lab02 lab03 lab04 lab05 lab06 lab07 lab08 lab09 lab10 \
        lab11 lab12 test test-cpu test-gpu test-lab01-cpu test-lab01 test-lab02 test-lab03 test-lab04 \
        test-lab05 test-lab06 test-lab07 test-lab08 test-lab09 test-lab10 test-lab11 test-lab12 sanitize \
        sanitize-lab01 sanitize-lab02 sanitize-lab03 sanitize-lab04 sanitize-lab05 sanitize-lab06 \
        sanitize-lab07 sanitize-lab08 sanitize-lab09 sanitize-lab10 sanitize-lab11 sanitize-lab12 \
        profile-lab09 bench-lab01 bench-lab02 bench-lab03 bench-lab05 bench-lab06 bench-lab08 bench-lab09 \
        bench-lab10 bench-lab12

all: cpu cuda
cpu: $(CPU_BINS)
cuda: $(CUDA_BINS)

lab01: $(BUILD)/calibrate_cpu $(BUILD)/calibrate_cuda
lab02: $(BUILD)/calibrate_channels
lab03: $(BUILD)/window_stats
lab04: $(BUILD)/screen
lab05: $(BUILD)/image_ops
lab06: $(BUILD)/box_filter
lab07: $(BUILD)/score_images
lab08: $(BUILD)/reduce_compare
lab09: $(BUILD)/profile_inspect
lab10: $(BUILD)/stream_batches
lab11: $(BUILD)/inspect
lab12: $(BUILD)/release_bench

help:
	@echo "Build:    make [all] | cpu | cuda | lab01 ... lab12    (SOLUTION=1 builds solutions/)"
	@echo "Test:     make test-cpu | test-gpu | test | test-lab01 ... test-lab12"
	@echo "Sanitize: make sanitize | sanitize-lab01 ... sanitize-lab12"
	@echo "Measure:  make bench-lab01 | bench-lab02 | bench-lab03 | bench-lab05 | bench-lab06 | bench-lab08 | bench-lab09 | bench-lab10"
	@echo "Release:  make bench-lab12 [RELEASE_LIST=... RELEASE_SPLIT=... RELEASE_CONFIG=...]"
	@echo "Profile:  make profile-lab09    (Nsight Systems and Nsight Compute reports in RUN_DIR)"
	@echo "Options:  CUDA_ARCH=$(CUDA_ARCH) SOLUTION=$(SOLUTION) BUILD=$(BUILD) OPT=$(OPT) RUN_DIR=runs/<time>"
	@echo "GPU tests, sanitizers, and benchmarks must run inside a GPU allocation."

clean:
	rm -rf build build-solution $(BUILD)

# ------------------------------------------------------------------------------ build rules

# Rebuild everything when the compilers, flags, or kernel source tree change.
$(BUILD)/.config: FORCE
	@mkdir -p $(@D)
	@echo '$(CXX) $(CXXFLAGS) | $(NVCC) $(NVCCFLAGS) | $(KERNEL_ROOT)' > $@.new
	@if cmp -s $@.new $@; then rm -f $@.new; else mv $@.new $@; fi

FORCE:

$(BUILD)/calibrate_cpu: src/lab01/calibrate_cpu.cpp src/lab01/common.hpp $(BUILD)/.config
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD)/calibrate_cuda: src/lab01/calibrate_cuda.cu src/lab01/common.hpp $(BUILD)/.config
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Harness code (identical in both builds) always comes from src/.
$(OBJ)/harness/%.o: src/%.cu $(BUILD)/.config
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# The code you write: src/ by default, solutions/ with SOLUTION=1.
$(OBJ)/kernels/%.o: $(KERNEL_ROOT)/%.cu $(BUILD)/.config
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(OBJ)/harness/lab02/main.o $(OBJ)/kernels/lab02/kernels.o: $(LAB02_H)
$(OBJ)/harness/lab03/main.o $(OBJ)/kernels/lab03/kernels.o: $(LAB03_H)
$(OBJ)/harness/lab04/screen.o $(OBJ)/kernels/lab04/gpu_pipeline.o: $(LAB04_H)
$(OBJ)/harness/lab05/main.o $(OBJ)/kernels/lab05/kernels.o: $(LAB05_H)
$(OBJ)/harness/lab06/main.o $(OBJ)/kernels/lab06/kernels.o: $(LAB06_H)
$(OBJ)/harness/lab07/main.o $(OBJ)/kernels/lab07/gpu_inspect.o: $(LAB07_H)
$(OBJ)/harness/lab08/main.o $(OBJ)/kernels/lab08/reductions.o: $(LAB08_H)
$(OBJ)/harness/lab09/main.o $(OBJ)/kernels/lab09/inspector.o: $(LAB09_H)
$(OBJ)/harness/lab10/main.o $(OBJ)/harness/lab10/serial.o $(OBJ)/kernels/lab10/pipeline.o: $(LAB10_H)
$(OBJ)/harness/lab11/main.o $(OBJ)/kernels/lab11/app.o: $(LAB11_H)
$(OBJ)/harness/lab12/main.o $(OBJ)/kernels/lab12/cpu_parallel.o: $(LAB12_H)

$(BUILD)/calibrate_channels: $(OBJ)/harness/lab02/main.o $(OBJ)/kernels/lab02/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD)/window_stats: $(OBJ)/harness/lab03/main.o $(OBJ)/kernels/lab03/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Lab 04 reuses your Lab 02 calibration and Lab 03 statistics kernels.
$(BUILD)/screen: $(OBJ)/harness/lab04/screen.o $(OBJ)/kernels/lab04/gpu_pipeline.o \
                 $(OBJ)/kernels/lab02/kernels.o $(OBJ)/kernels/lab03/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD)/image_ops: $(OBJ)/harness/lab05/main.o $(OBJ)/kernels/lab05/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD)/box_filter: $(OBJ)/harness/lab06/main.o $(OBJ)/kernels/lab06/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Lab 07 reuses your Lab 05 pixel kernels and Lab 06 filter.
$(BUILD)/score_images: $(OBJ)/harness/lab07/main.o $(OBJ)/kernels/lab07/gpu_inspect.o \
                       $(OBJ)/kernels/lab05/kernels.o $(OBJ)/kernels/lab06/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD)/reduce_compare: $(OBJ)/harness/lab08/main.o $(OBJ)/kernels/lab08/reductions.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Lab 09 compares your buffer-reusing Inspector with your Lab 07 baseline (and its Lab 05/06 kernels).
$(BUILD)/profile_inspect: $(OBJ)/harness/lab09/main.o $(OBJ)/kernels/lab09/inspector.o \
                          $(OBJ)/kernels/lab07/gpu_inspect.o $(OBJ)/kernels/lab05/kernels.o \
                          $(OBJ)/kernels/lab06/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Lab 10's provided serial baseline (src/lab10/serial.cu) is harness code; your pipeline is not.
$(BUILD)/stream_batches: $(OBJ)/harness/lab10/main.o $(OBJ)/harness/lab10/serial.o \
                         $(OBJ)/kernels/lab10/pipeline.o $(OBJ)/kernels/lab02/kernels.o \
                         $(OBJ)/kernels/lab03/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# The capstone (Lab 11) integrates your Labs 02-07 and 09 into one application.
$(BUILD)/inspect: $(OBJ)/harness/lab11/main.o $(OBJ)/kernels/lab11/app.o $(OBJ)/kernels/lab09/inspector.o \
                  $(OBJ)/kernels/lab07/gpu_inspect.o $(OBJ)/kernels/lab05/kernels.o $(OBJ)/kernels/lab06/kernels.o \
                  $(OBJ)/kernels/lab04/gpu_pipeline.o $(OBJ)/kernels/lab02/kernels.o $(OBJ)/kernels/lab03/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Lab 12 measures your multicore CPU baseline against your Lab 07 and Lab 09 GPU versions.
$(BUILD)/release_bench: $(OBJ)/harness/lab12/main.o $(OBJ)/kernels/lab12/cpu_parallel.o \
                        $(OBJ)/kernels/lab09/inspector.o $(OBJ)/kernels/lab07/gpu_inspect.o \
                        $(OBJ)/kernels/lab05/kernels.o $(OBJ)/kernels/lab06/kernels.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# ------------------------------------------------------------------------------ tests
# scripts/check.sh prints PASS/FAIL per command; `fail` expects a clean error exit (1 or 2).
# Use `make -k ...` to keep going after the first failure.

test: test-cpu test-gpu
test-cpu: test-lab01-cpu
test-gpu: test-lab01 test-lab02 test-lab03 test-lab04 test-lab05 test-lab06 test-lab07 test-lab08 \
          test-lab09 test-lab10 test-lab11 test-lab12

test-lab01-cpu: $(BUILD)/calibrate_cpu
	@for n in 1 255 256 257 1000003; do $(CHECK) pass $(BUILD)/calibrate_cpu $$n || exit 1; done
	@for v in 0 -1 abc 12x 100000001; do $(CHECK) fail $(BUILD)/calibrate_cpu $$v || exit 1; done

test-lab01: test-lab01-cpu $(BUILD)/calibrate_cuda
	@for n in 1 255 256 257 1000003; do $(CHECK) pass $(BUILD)/calibrate_cuda $$n || exit 1; done

L02 := $(BUILD)/calibrate_channels --reps 2 --warmups 1 --header 0
test-lab02: $(BUILD)/calibrate_channels
	@for layout in channel sample; do \
	  for shape in "1 1" "1 257" "3 255" "3 256" "32 257" "32 100003" "1000 1"; do \
	    set -- $$shape; \
	    $(CHECK) pass $(L02) --layout $$layout --channels $$1 --samples $$2 || exit 1; \
	  done; \
	  $(CHECK) pass $(L02) --layout $$layout --channels 5 --samples 300001 --threads 128 --blocks 3 || exit 1; \
	  $(CHECK) pass $(L02) --layout $$layout --channels 64 --samples 1000 --threads 1024 || exit 1; \
	done
	@$(CHECK) fail $(L02) --channels 0
	@$(CHECK) fail $(L02) --samples 12x
	@$(CHECK) fail $(L02) --threads 100
	@$(CHECK) fail $(L02) --layout diagonal
	@$(CHECK) fail $(L02) --channels 20000 --samples 10000
	@$(CHECK) fail $(L02) --low 5 --high 5
	@$(CHECK) fail $(L02) --colour blue

L03 := $(BUILD)/window_stats --reps 2 --warmups 1 --header 0
test-lab03: $(BUILD)/window_stats
	@for mode in fused separate; do \
	  for pattern in zeros constant alternating impulse ramp vibration; do \
	    for shape in "1 1" "1000 1" "1000 7" "1000 256" "777 1000" "5000 1000" "100003 4096" "20000 20000"; do \
	      set -- $$shape; \
	      $(CHECK) pass $(L03) --mode $$mode --pattern $$pattern --samples $$1 --window $$2 || exit 1; \
	    done; \
	  done; \
	  for threads in 32 1024; do \
	    $(CHECK) pass $(L03) --mode $$mode --threads $$threads --samples 100000 --window 3000 || exit 1; \
	  done; \
	done
	@$(CHECK) fail $(L03) --window 0
	@$(CHECK) fail $(L03) --samples 0
	@$(CHECK) fail $(L03) --threads 96
	@$(CHECK) fail $(L03) --pattern sawtooth
	@$(CHECK) fail $(L03) --mode both

L04_DATA := $(BUILD)/lab04-test
test-lab04: $(BUILD)/screen
	@rm -rf $(L04_DATA) && mkdir -p $(L04_DATA)/reports
	@$(PYTHON) scripts/lab04_synthetic.py --output $(L04_DATA) --seconds 2 > /dev/null
	@for f in $(L04_DATA)/*.f32; do \
	  id=$$(basename $$f .f32); \
	  $(CHECK) pass $(BUILD)/screen --input $$f --window 2048 --backend both --crest-threshold 3.6 \
	    --output $(L04_DATA)/reports/$$id.csv || exit 1; \
	done
	@for f in $(L04_DATA)/edge/*.f32; do \
	  for window in 1 4 2048; do \
	    $(CHECK) pass $(BUILD)/screen --input $$f --window $$window --backend both \
	      --output $(L04_DATA)/edge/report.csv || exit 1; \
	  done; \
	done
	@$(CHECK) pass $(PYTHON) scripts/lab04_evaluate.py sweep --labels $(L04_DATA)/labels.csv \
	  --feature crest $(L04_DATA)/reports/*.csv
	@$(CHECK) pass $(PYTHON) scripts/lab04_evaluate.py evaluate --labels $(L04_DATA)/labels.csv \
	  --split test $(L04_DATA)/reports/*.csv
	@printf 'abc' > $(L04_DATA)/truncated.bin
	@$(CHECK) fail $(BUILD)/screen --input $(L04_DATA)/truncated.bin --backend cpu --output $(L04_DATA)/x.csv
	@$(CHECK) fail $(BUILD)/screen --input $(L04_DATA)/missing.f32 --backend cpu --output $(L04_DATA)/x.csv
	@$(CHECK) fail $(BUILD)/screen --input $(L04_DATA)/syn-normal-01.f32 --window 0 --output $(L04_DATA)/x.csv
	@$(CHECK) fail $(BUILD)/screen --input $(L04_DATA)/syn-normal-01.f32 --backend tpu --output $(L04_DATA)/x.csv

L05 := $(BUILD)/image_ops --reps 2 --warmups 1 --header 0
test-lab05: $(BUILD)/image_ops
	@for block in 16x16 32x8 8x32; do \
	  for size in "1 1" "1 37" "37 1" "641 479" "64 64" "1920 1080"; do \
	    set -- $$size; \
	    for channels in 1 3; do \
	      $(CHECK) pass $(L05) --width $$1 --height $$2 --channels $$channels --block $$block || exit 1; \
	    done; \
	  done; \
	done
	@for pattern in gradient constant checker noise; do \
	  $(CHECK) pass $(L05) --width 641 --height 479 --pattern $$pattern --host-pad 13 || exit 1; \
	done
	@$(CHECK) pass $(L05) --width 641 --height 479 --pattern constant --check-defect 1
	@$(CHECK) pass $(L05) --width 641 --height 479 --channels 1 --pattern constant --check-defect 1
	@$(CHECK) fail $(L05) --width 0
	@$(CHECK) fail $(L05) --channels 2
	@$(CHECK) fail $(L05) --block 64x32
	@$(CHECK) fail $(L05) --block 16
	@$(CHECK) fail $(L05) --lo 200 --hi 100

L06 := $(BUILD)/box_filter --reps 2 --warmups 1 --cpu-reps 1 --header 0
test-lab06: $(BUILD)/box_filter
	@for variant in direct tiled; do \
	  for radius in 1 2 3 7; do \
	    for case in "1 1 impulse" "2 3 noise" "5 5 corner" "33 17 constant" "641 479 noise" "256 256 impulse"; do \
	      set -- $$case; \
	      $(CHECK) pass $(L06) --variant $$variant --radius $$radius --width $$1 --height $$2 --pattern $$3 || exit 1; \
	    done; \
	  done; \
	  for block in 32x8 8x32 32x32; do \
	    $(CHECK) pass $(L06) --variant $$variant --block $$block --radius 2 --width 641 --height 479 --host-pad 7 || exit 1; \
	  done; \
	done
	@$(CHECK) fail $(L06) --radius 33
	@$(CHECK) fail $(L06) --variant fft
	@$(CHECK) fail $(L06) --block 0x8

L07_DATA := $(BUILD)/lab07-test
L07 := $(BUILD)/score_images --header 0
test-lab07: $(BUILD)/score_images
	@rm -rf $(L07_DATA) && mkdir -p $(L07_DATA)/masks
	@$(PYTHON) scripts/lab07_synthetic.py --output $(L07_DATA) > /dev/null
	@for split in train dev test; do \
	  $(CHECK) pass $(L07) --list $(L07_DATA)/images.csv --split $$split --backend both \
	    --output $(L07_DATA)/$$split-scores.csv || exit 1; \
	done
	@for block in 32x8 8x32 32x32; do \
	  for radius in 0 1 5; do \
	    $(CHECK) pass $(L07) --list $(L07_DATA)/edge/images.csv --backend both --block $$block --radius $$radius \
	      --output $(L07_DATA)/edge-scores.csv || exit 1; \
	  done; \
	done
	@$(CHECK) pass $(L07) --list $(L07_DATA)/images.csv --split test --backend both --image-threshold 0.001 \
	  --save-masks $(L07_DATA)/masks --output $(L07_DATA)/test-predicted.csv
	@$(CHECK) pass $(PYTHON) scripts/lab07_evaluate.py calibrate --scores $(L07_DATA)/dev-scores.csv
	@$(CHECK) pass $(PYTHON) scripts/lab07_evaluate.py evaluate --scores $(L07_DATA)/test-predicted.csv --threshold 0.001
	@$(CHECK) fail $(L07) --list $(L07_DATA)/edge/bad.csv --output $(L07_DATA)/x.csv
	@$(CHECK) fail $(L07) --list $(L07_DATA)/missing.csv --output $(L07_DATA)/x.csv
	@$(CHECK) fail $(L07) --list $(L07_DATA)/images.csv --split validation --output $(L07_DATA)/x.csv
	@$(CHECK) fail $(L07) --list $(L07_DATA)/images.csv --block 24x16 --output $(L07_DATA)/x.csv
	@$(CHECK) fail $(L07) --list $(L07_DATA)/images.csv --radius 17 --output $(L07_DATA)/x.csv

L08 := $(BUILD)/reduce_compare --reps 2 --warmups 1 --cpu-reps 1 --header 0
test-lab08: $(BUILD)/reduce_compare
	@for backend in custom thrust cub; do \
	  for pattern in zeros ones sparse dense random values; do \
	    for bytes in 1 255 256 257 1000003; do \
	      $(CHECK) pass $(L08) --backend $$backend --pattern $$pattern --bytes $$bytes || exit 1; \
	    done; \
	  done; \
	  $(CHECK) pass $(L08) --backend $$backend --pattern values --bytes 100000007 --threads 1024 || exit 1; \
	done
	@$(CHECK) pass $(L08) --backend all --pattern random --bytes 4096 --threads 32
	@$(CHECK) fail $(L08) --bytes 0
	@$(CHECK) fail $(L08) --backend magic
	@$(CHECK) fail $(L08) --threads 96

L09_DATA := $(BUILD)/lab09-test
L09 := $(BUILD)/profile_inspect --header 0
test-lab09: $(BUILD)/profile_inspect
	@rm -rf $(L09_DATA) && mkdir -p $(L09_DATA)
	@$(PYTHON) scripts/lab07_synthetic.py --output $(L09_DATA) --size 64 > /dev/null
	@$(CHECK) pass $(L09) --list $(L09_DATA)/images.csv --variant both --repeat 2
	@for block in 16x16 32x8; do \
	  $(CHECK) pass $(L09) --list $(L09_DATA)/edge/images.csv --variant both --repeat 3 --block $$block || exit 1; \
	done
	@$(CHECK) pass $(L09) --list $(L09_DATA)/edge/images.csv --variant reuse --repeat 1 --radius 0
	@$(CHECK) pass $(L09) --list $(L09_DATA)/images.csv --split dev --variant reuse --repeat 4
	@$(CHECK) fail $(L09) --list $(L09_DATA)/images.csv --variant turbo
	@$(CHECK) fail $(L09) --list $(L09_DATA)/images.csv --repeat 0
	@$(CHECK) fail $(L09) --list $(L09_DATA)/edge/bad.csv

L10 := $(BUILD)/stream_batches --header 0
test-lab10: $(BUILD)/stream_batches
	@for batch in 1 3 4 7 16 64; do \
	  $(CHECK) pass $(L10) --mode both --items 37 --item-samples 8192 --window 512 --batch $$batch || exit 1; \
	done
	@$(CHECK) pass $(L10) --mode both --items 1 --item-samples 1024 --window 1024 --batch 4
	@$(CHECK) pass $(L10) --mode both --items 64 --item-samples 65536 --window 1024 --batch 16 --threads 1024
	@$(CHECK) pass $(L10) --mode both --items 24 --item-samples 4096 --window 256 --batch 4 --interval-ms 0.2
	@$(CHECK) pass $(L10) --mode streams --items 9 --item-samples 2048 --window 2048 --batch 2 --threads 32
	@$(CHECK) fail $(L10) --item-samples 1000 --window 256
	@$(CHECK) fail $(L10) --batch 0
	@$(CHECK) fail $(L10) --mode async
	@$(CHECK) fail $(L10) --interval-ms -1

# The capstone tests compare inspect with the single-lab tools it integrates (score_images, screen).
L11_DATA := $(BUILD)/lab11-test
INSPECT := $(BUILD)/inspect
test-lab11: $(BUILD)/inspect $(BUILD)/score_images $(BUILD)/screen
	@rm -rf $(L11_DATA) && mkdir -p $(L11_DATA)/one $(L11_DATA)/masks $(L11_DATA)/bad $(L11_DATA)/empty
	@$(PYTHON) scripts/lab07_synthetic.py --output $(L11_DATA)/images --size 64 > /dev/null
	@$(PYTHON) scripts/lab04_synthetic.py --output $(L11_DATA)/sensors --seconds 1 > /dev/null
	@cp $(L11_DATA)/images/syn-test-scratch-00.pgm $(L11_DATA)/one/
	@printf 'P2\n1 1\n255\n0\n' > $(L11_DATA)/bad/ascii.pgm
	@printf 'radiuss = 3\n' > $(L11_DATA)/typo.conf
	@for backend in cpu cuda both; do \
	  $(CHECK) pass $(INSPECT) images --header 0 --config configs/synthetic-images.conf --input $(L11_DATA)/images \
	    --backend $$backend --output $(L11_DATA)/images-$$backend.csv || exit 1; \
	done
	@$(CHECK) pass $(PYTHON) scripts/compare_reports.py same $(L11_DATA)/images-cpu.csv $(L11_DATA)/images-cuda.csv \
	  --key image_id --columns mask_pixels,score,predicted_defect
	@$(CHECK) pass $(PYTHON) scripts/compare_reports.py threshold $(L11_DATA)/images-cuda.csv --threshold 0
	@$(CHECK) pass $(BUILD)/score_images --header 0 --list $(L11_DATA)/images/images.csv --image-threshold 0 \
	  --output $(L11_DATA)/score-images.csv
	@$(CHECK) pass $(PYTHON) scripts/compare_reports.py same $(L11_DATA)/images-cuda.csv $(L11_DATA)/score-images.csv \
	  --key image_id --columns mask_pixels,score,predicted_defect
	@$(CHECK) pass $(PYTHON) scripts/compare_reports.py record $(L11_DATA)/images-cuda.csv.json mode=images \
	  backend=cuda settings.radius=3 settings.pixel_threshold=30 settings.image_threshold=0 summary.disagreements=0
	@$(CHECK) pass $(INSPECT) images --header 0 --config configs/synthetic-images.conf \
	  --input $(L11_DATA)/images/images.csv --split test --output $(L11_DATA)/images-list.csv
	@$(CHECK) pass $(INSPECT) images --header 0 --config configs/synthetic-images.conf --input $(L11_DATA)/one \
	  --save-masks $(L11_DATA)/masks --output $(L11_DATA)/images-one.csv
	@$(CHECK) pass test -s $(L11_DATA)/masks/syn-test-scratch-00-mask.pgm
	@$(CHECK) pass $(INSPECT) images --header 0 --input $(L11_DATA)/images/edge/edge-rgb-scratch.ppm --backend both \
	  --radius 1 --output $(L11_DATA)/images-single.csv
	@for backend in cpu cuda both; do \
	  $(CHECK) pass $(INSPECT) sensors --header 0 --config configs/synthetic-sensors.conf \
	    --input $(L11_DATA)/sensors/syn-fault-01.f32 --backend $$backend --output $(L11_DATA)/sensors-$$backend.csv || exit 1; \
	done
	@$(CHECK) pass $(BUILD)/screen --header 0 --input $(L11_DATA)/sensors/syn-fault-01.f32 --window 2048 \
	  --crest-threshold 3.6 --output $(L11_DATA)/screen.csv
	@$(CHECK) pass $(PYTHON) scripts/compare_reports.py same $(L11_DATA)/sensors-cuda.csv $(L11_DATA)/screen.csv \
	  --key window_start --columns mean,rms,peak,alarm
	@$(CHECK) pass $(PYTHON) scripts/compare_reports.py close $(L11_DATA)/sensors-cpu.csv $(L11_DATA)/sensors-cuda.csv \
	  --key window_start --columns mean,rms,peak --rel 1e-4 --abs 1e-5
	@$(CHECK) fail $(INSPECT) images --input $(L11_DATA)/bad --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT) images --input $(L11_DATA)/empty --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT) images --input $(L11_DATA)/missing --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT) images --config $(L11_DATA)/typo.conf --input $(L11_DATA)/images --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT) images --config configs/synthetic-images.conf --radius 99 --input $(L11_DATA)/images \
	  --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT) images --config configs/synthetic-sensors.conf --input $(L11_DATA)/images \
	  --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT) sensors --input $(L11_DATA)/sensors/missing.f32 --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT) everything --input $(L11_DATA)/images --output $(L11_DATA)/x.csv
	@$(CHECK) fail $(INSPECT)

L12_DATA := $(BUILD)/lab12-test
RELEASE := $(BUILD)/release_bench --header 0
test-lab12: $(BUILD)/release_bench
	@rm -rf $(L12_DATA) && mkdir -p $(L12_DATA)
	@$(PYTHON) scripts/lab07_synthetic.py --output $(L12_DATA)/images --size 48 > /dev/null
	@$(CHECK) pass $(RELEASE) --list $(L12_DATA)/images/images.csv --config configs/synthetic-images.conf \
	  --output-dir $(L12_DATA)/run --sizes 1,5,all --repeats 2 --warmups 1 --cpu-threads 3
	@$(CHECK) pass $(PYTHON) scripts/lab12_report.py $(L12_DATA)/run
	@$(CHECK) pass test -s $(L12_DATA)/run/report.md
	@$(CHECK) pass $(RELEASE) --list $(L12_DATA)/images/images.csv --split test --variants cpu,cpu-parallel \
	  --cpu-threads 64 --repeats 1 --warmups 0 --output-dir $(L12_DATA)/run-cpu
	@$(CHECK) pass $(RELEASE) --list $(L12_DATA)/images/edge/images.csv --sizes all --repeats 1 \
	  --output-dir $(L12_DATA)/run-edge
	@$(CHECK) fail $(RELEASE) --list $(L12_DATA)/images/images.csv --variants cpu,gpu-magic --output-dir $(L12_DATA)/x
	@$(CHECK) fail $(RELEASE) --list $(L12_DATA)/images/images.csv --sizes 0 --output-dir $(L12_DATA)/x
	@$(CHECK) fail $(RELEASE) --list $(L12_DATA)/images/images.csv --sizes 1,many --output-dir $(L12_DATA)/x
	@$(CHECK) fail $(RELEASE) --list $(L12_DATA)/images/images.csv

# ------------------------------------------------------------------------------ sanitizers
# Small inputs: the tools slow execution down a lot. Never time a sanitized run.

SAN := --reps 1 --warmups 0 --header 0
sanitize: sanitize-lab01 sanitize-lab02 sanitize-lab03 sanitize-lab04 sanitize-lab05 sanitize-lab06 \
          sanitize-lab07 sanitize-lab08 sanitize-lab09 sanitize-lab10 sanitize-lab11 sanitize-lab12

sanitize-lab01: $(BUILD)/calibrate_cuda
	@for n in 1 257 1000003; do $(CHECK) pass $(MEMCHECK) $(BUILD)/calibrate_cuda $$n || exit 1; done

sanitize-lab02: $(BUILD)/calibrate_channels
	@for layout in channel sample; do \
	  $(CHECK) pass $(MEMCHECK) $(BUILD)/calibrate_channels $(SAN) --layout $$layout --channels 3 --samples 1001 || exit 1; \
	  $(CHECK) pass $(MEMCHECK) $(BUILD)/calibrate_channels $(SAN) --layout $$layout --channels 3 --samples 1001 --blocks 2 || exit 1; \
	done

sanitize-lab03: $(BUILD)/window_stats
	@for mode in fused separate; do \
	  for tool in "$(MEMCHECK)" "$(RACECHECK)" "$(SYNCCHECK)"; do \
	    $(CHECK) pass $$tool $(BUILD)/window_stats $(SAN) --mode $$mode --samples 3000 --window 1000 --threads 128 || exit 1; \
	  done; \
	done

sanitize-lab04: $(BUILD)/screen
	@rm -rf $(BUILD)/lab04-sanitize && mkdir -p $(BUILD)/lab04-sanitize
	@$(PYTHON) scripts/lab04_synthetic.py --output $(BUILD)/lab04-sanitize --seconds 0.2 > /dev/null
	@$(CHECK) pass $(MEMCHECK) $(BUILD)/screen --input $(BUILD)/lab04-sanitize/syn-fault-01.f32 \
	  --window 512 --backend cuda --output $(BUILD)/lab04-sanitize/report.csv

sanitize-lab05: $(BUILD)/image_ops
	@for channels in 1 3; do \
	  $(CHECK) pass $(MEMCHECK) $(BUILD)/image_ops $(SAN) --width 37 --height 29 --channels $$channels --host-pad 5 || exit 1; \
	done

sanitize-lab06: $(BUILD)/box_filter
	@for variant in direct tiled; do \
	  for tool in "$(MEMCHECK)" "$(RACECHECK)" "$(SYNCCHECK)"; do \
	    $(CHECK) pass $$tool $(BUILD)/box_filter $(SAN) --cpu-reps 1 --variant $$variant --radius 2 --width 41 --height 23 || exit 1; \
	  done; \
	done

sanitize-lab07: $(BUILD)/score_images
	@rm -rf $(BUILD)/lab07-sanitize && mkdir -p $(BUILD)/lab07-sanitize
	@$(PYTHON) scripts/lab07_synthetic.py --output $(BUILD)/lab07-sanitize --size 24 > /dev/null
	@for tool in "$(MEMCHECK)" "$(RACECHECK)" "$(SYNCCHECK)"; do \
	  $(CHECK) pass $$tool $(BUILD)/score_images --header 0 --list $(BUILD)/lab07-sanitize/edge/images.csv \
	    --backend cuda --output $(BUILD)/lab07-sanitize/scores.csv || exit 1; \
	done

sanitize-lab08: $(BUILD)/reduce_compare
	@for backend in custom thrust cub; do \
	  $(CHECK) pass $(MEMCHECK) $(BUILD)/reduce_compare $(SAN) --cpu-reps 1 --backend $$backend --bytes 100003 || exit 1; \
	done
	@for tool in "$(RACECHECK)" "$(SYNCCHECK)"; do \
	  $(CHECK) pass $$tool $(BUILD)/reduce_compare $(SAN) --cpu-reps 1 --backend custom --bytes 100003 || exit 1; \
	done

sanitize-lab09: $(BUILD)/profile_inspect
	@rm -rf $(BUILD)/lab09-sanitize && mkdir -p $(BUILD)/lab09-sanitize
	@$(PYTHON) scripts/lab07_synthetic.py --output $(BUILD)/lab09-sanitize --size 24 > /dev/null
	@for tool in "$(MEMCHECK)" "$(RACECHECK)" "$(SYNCCHECK)"; do \
	  $(CHECK) pass $$tool $(BUILD)/profile_inspect --header 0 --list $(BUILD)/lab09-sanitize/edge/images.csv \
	    --variant both --repeat 2 || exit 1; \
	done

sanitize-lab10: $(BUILD)/stream_batches
	@for tool in "$(MEMCHECK)" "$(RACECHECK)" "$(SYNCCHECK)"; do \
	  $(CHECK) pass $$tool $(BUILD)/stream_batches --header 0 --mode both --items 9 --item-samples 2048 \
	    --window 256 --batch 2 || exit 1; \
	done

sanitize-lab11: $(BUILD)/inspect
	@rm -rf $(BUILD)/lab11-sanitize && mkdir -p $(BUILD)/lab11-sanitize
	@$(PYTHON) scripts/lab07_synthetic.py --output $(BUILD)/lab11-sanitize/images --size 24 > /dev/null
	@$(PYTHON) scripts/lab04_synthetic.py --output $(BUILD)/lab11-sanitize/sensors --seconds 0.2 > /dev/null
	@$(CHECK) pass $(MEMCHECK) $(BUILD)/inspect images --header 0 --input $(BUILD)/lab11-sanitize/images/edge/images.csv \
	  --backend cuda --output $(BUILD)/lab11-sanitize/images.csv
	@$(CHECK) pass $(MEMCHECK) $(BUILD)/inspect sensors --header 0 --input $(BUILD)/lab11-sanitize/sensors/syn-fault-01.f32 \
	  --window 512 --backend cuda --output $(BUILD)/lab11-sanitize/sensors.csv

sanitize-lab12: $(BUILD)/release_bench
	@rm -rf $(BUILD)/lab12-sanitize && mkdir -p $(BUILD)/lab12-sanitize
	@$(PYTHON) scripts/lab07_synthetic.py --output $(BUILD)/lab12-sanitize/images --size 24 > /dev/null
	@$(CHECK) pass $(MEMCHECK) $(BUILD)/release_bench --header 0 --list $(BUILD)/lab12-sanitize/images/edge/images.csv \
	  --variants cuda-baseline,cuda --sizes all --repeats 1 --warmups 0 --output-dir $(BUILD)/lab12-sanitize/run

# ------------------------------------------------------------------------------ profiling
# Lab 09 captures, inside a GPU allocation. Reports land in $(RUN_DIR) (*.nsys-rep and *.ncu-rep are
# ignored by Git). If Nsight Compute reports ERR_NVGPUCTRPERM, see docs/sol-setup.md.

profile-lab09: $(BUILD)/profile_inspect $(BUILD)/box_filter
	@mkdir -p $(RUN_DIR)
	@$(PYTHON) scripts/lab07_synthetic.py --output $(RUN_DIR)/images --size 384 > /dev/null
	@for variant in baseline reuse; do \
	  $(NSYS) profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=$(RUN_DIR)/lab09-$$variant \
	    $(BUILD)/profile_inspect --list $(RUN_DIR)/images/images.csv --variant $$variant --verify 0 --repeat 5 || exit 1; \
	  $(NSYS) stats $(RUN_DIR)/lab09-$$variant.nsys-rep > $(RUN_DIR)/lab09-$$variant-stats.txt || exit 1; \
	done
	@for variant in direct tiled; do \
	  $(NCU) --set basic --kernel-name regex:box_ --launch-count 1 --force-overwrite \
	    --export $(RUN_DIR)/lab09-box-$$variant $(BUILD)/box_filter --variant $$variant --radius 3 \
	    --width 1920 --height 1080 --reps 1 --warmups 0 --cpu-reps 1 || exit 1; \
	done
	@echo "Reports in $(RUN_DIR): open them in Nsight Systems and Nsight Compute, or read lab09-*-stats.txt."

# ------------------------------------------------------------------------------ experiments
# Each writes CSV files under $(RUN_DIR) (default runs/<timestamp>/, ignored by Git).

bench-lab01: $(BUILD)/calibrate_cpu $(BUILD)/calibrate_cuda
	@mkdir -p $(RUN_DIR)
	$(BUILD)/calibrate_cpu 1000003 $(RUN_DIR)/lab01-cpu-raw.csv > $(RUN_DIR)/lab01-cpu.csv
	$(BUILD)/calibrate_cuda 1000003 $(RUN_DIR)/lab01-cuda-raw.csv > $(RUN_DIR)/lab01-cuda.csv
	@echo "Wrote $(RUN_DIR)/lab01-*.csv"

bench-lab02: $(BUILD)/calibrate_channels
	@mkdir -p $(RUN_DIR)
	@h=1; for layout in channel sample; do for threads in 128 256 512; do \
	  $(BUILD)/calibrate_channels --layout $$layout --channels 64 --samples 1000000 --threads $$threads --header $$h || exit 1; \
	  h=0; \
	done; done > $(RUN_DIR)/lab02.csv
	@echo "Wrote $(RUN_DIR)/lab02.csv"

bench-lab03: $(BUILD)/window_stats
	@mkdir -p $(RUN_DIR)
	@h=1; for mode in fused separate; do for window in 256 4096 65536; do for threads in 128 256 512; do \
	  $(BUILD)/window_stats --mode $$mode --samples 16777216 --window $$window --threads $$threads --header $$h || exit 1; \
	  h=0; \
	done; done; done > $(RUN_DIR)/lab03.csv
	@echo "Wrote $(RUN_DIR)/lab03.csv"

bench-lab05: $(BUILD)/image_ops
	@mkdir -p $(RUN_DIR)
	@h=1; for size in "1920 1080" "3840 2160"; do set -- $$size; for block in 8x8 16x16 32x8 8x32 32x32; do \
	  $(BUILD)/image_ops --width $$1 --height $$2 --block $$block --header $$h || exit 1; \
	  h=0; \
	done; done > $(RUN_DIR)/lab05.csv
	@echo "Wrote $(RUN_DIR)/lab05.csv"

bench-lab06: $(BUILD)/box_filter
	@mkdir -p $(RUN_DIR)
	@h=1; for size in "1920 1080" "3840 2160"; do set -- $$size; for radius in 1 2 3 5 7; do for variant in direct tiled; do \
	  $(BUILD)/box_filter --variant $$variant --radius $$radius --width $$1 --height $$2 --cpu-reps 1 --header $$h || exit 1; \
	  h=0; \
	done; done; done > $(RUN_DIR)/lab06.csv
	@echo "Wrote $(RUN_DIR)/lab06.csv"

# One process per backend, so each first_ms includes only that backend's own startup.
bench-lab08: $(BUILD)/reduce_compare
	@mkdir -p $(RUN_DIR)
	@h=1; for bytes in 1000 100000 10000000 1000000000; do for backend in custom thrust cub; do \
	  $(BUILD)/reduce_compare --backend $$backend --pattern random --bytes $$bytes --cpu-reps 1 --header $$h || exit 1; \
	  h=0; \
	done; done > $(RUN_DIR)/lab08.csv
	@echo "Wrote $(RUN_DIR)/lab08.csv"

bench-lab09: $(BUILD)/profile_inspect
	@mkdir -p $(RUN_DIR)
	@$(PYTHON) scripts/lab07_synthetic.py --output $(RUN_DIR)/images --size 384 > /dev/null
	@$(BUILD)/profile_inspect --list $(RUN_DIR)/images/images.csv --variant both --repeat 5 > $(RUN_DIR)/lab09.csv
	@echo "Wrote $(RUN_DIR)/lab09.csv"

# Total work fixed (256 items); offline (all items available) and a simulated arrival every 0.05 ms.
bench-lab10: $(BUILD)/stream_batches
	@mkdir -p $(RUN_DIR)
	@h=1; for interval in 0 0.05; do for batch in 1 4 16 64; do \
	  $(BUILD)/stream_batches --mode both --items 256 --batch $$batch --interval-ms $$interval --header $$h || exit 1; \
	  h=0; \
	done; done > $(RUN_DIR)/lab10.csv
	@echo "Wrote $(RUN_DIR)/lab10.csv"

# Release measurements (Lab 12). With no RELEASE_LIST, a synthetic 512x512 set is generated; for the
# release, point it at your frozen dataset, e.g.
#   make bench-lab12 RELEASE_LIST=data/processed/mvtec-bottle/images.csv RELEASE_SPLIT=test \
#        RELEASE_CONFIG=configs/mvtec-bottle.conf
RELEASE_LIST   ?=
RELEASE_SPLIT  ?= all
RELEASE_CONFIG ?= configs/synthetic-images.conf
bench-lab12: $(BUILD)/release_bench
	@mkdir -p $(RUN_DIR)
	@if [ -z "$(RELEASE_LIST)" ]; then $(PYTHON) scripts/lab07_synthetic.py --output $(RUN_DIR)/images --size 512 > /dev/null; fi
	SOURCE_COMMIT=$$(git rev-parse HEAD 2>/dev/null || echo unknown) $(BUILD)/release_bench \
	  --list $(if $(RELEASE_LIST),$(RELEASE_LIST),$(RUN_DIR)/images/images.csv) --split $(RELEASE_SPLIT) \
	  --config $(RELEASE_CONFIG) --sizes 1,10,all --repeats 10 --warmups 2 --output-dir $(RUN_DIR)/release
	$(PYTHON) scripts/lab12_report.py $(RUN_DIR)/release
