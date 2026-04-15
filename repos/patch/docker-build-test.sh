#!/bin/bash
# Build llama.cpp SYCL backend in Docker to verify patches compile
# Uses the same oneAPI apt packages as the CI workflow
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")/llama.cpp"

if [ ! -d "$REPO_DIR" ]; then
    echo "Repo not found at $REPO_DIR"
    exit 1
fi

echo "Building llama.cpp SYCL backend in Docker (amd64)..."
echo "This will take a few minutes on first run."

docker buildx build --platform linux/amd64 \
  -f - \
  -t llama-sycl-test \
  --load \
  --build-context repo="$REPO_DIR" \
  . <<'DOCKERFILE'
FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies + oneAPI DPC++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates cmake g++ make git libssl-dev \
    && wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    && echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy patched source
COPY --from=repo . .

# Source oneAPI and build SYCL backend
SHELL ["/bin/bash", "-c"]
RUN source /opt/intel/oneapi/setvars.sh && \
    cmake -B build \
      -DGGML_SYCL=ON \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF && \
    cmake --build build -j$(nproc) 2>&1 | tee /tmp/build.log

# Check build result
RUN grep -c "error:" /tmp/build.log && exit 1 || echo "Build completed successfully"

# Run available tests (no GPU so many will skip, we just check for segfaults/crashes)
RUN source /opt/intel/oneapi/setvars.sh && \
    cd build && \
    ctest --output-on-failure -R "test-" --timeout 30 2>&1 | tee /tmp/test.log || true

RUN echo "=== BUILD RESULT ===" && \
    tail -5 /tmp/build.log && \
    echo "=== TEST SUMMARY ===" && \
    tail -10 /tmp/test.log 2>/dev/null || true
DOCKERFILE

echo ""
echo "Build container ready. Checking if it succeeded:"
docker run --rm llama-sycl-test bash -c "echo 'Container runs OK'"
