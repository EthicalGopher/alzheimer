# syntax=docker/dockerfile:1.4

# ===================================================================
# Build Stage: Compiles the Go application with all dependencies
# ===================================================================
FROM ubuntu:latest AS build-dev

# Install dependencies, add Coral repo, and install EdgeTPU dev library in fewer layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    gcc-x86-64-linux-gnu \
    golang-1.21 \
    curl \
    gnupg \
    ca-certificates && \
    # Add Coral GPG key and repository using the modern, recommended method
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list > /dev/null && \
    # Update apt lists again to include the new repo, then install the dev library
    apt-get update && \
    apt-get install -y --no-install-recommends libedgetpu1-dev && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Set up Go environment
WORKDIR /go/src/app
RUN ln -s /usr/lib/go-1.21/bin/go /usr/bin/go

# Download and install the TFLite C library
RUN wget https://github.com/mattn/go-tflite/releases/download/v1.0.5/go-tflite-buildkit-20240529.tar.gz && \
    tar -C /usr/local -xvf go-tflite-buildkit-20240529.tar.gz && \
    rm go-tflite-buildkit-20240529.tar.gz

# Download Go modules before copying source to leverage layer caching
COPY --link go.mod go.sum ./
RUN go mod download

# Copy source and build the application
COPY --link . .
RUN CGO_ENABLED=1 go build -buildvcs=false -trimpath -ldflags '-w -s' -o /app/alzheimer


# ===================================================================
# Final Stage: Creates a minimal image with only the runtime essentials
# ===================================================================
FROM ubuntu:latest

# Install runtime dependencies, add Coral repo, and install EdgeTPU runtime library
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates && \
    # Add Coral GPG key and repository
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list > /dev/null && \
    # Update and install runtime library
    apt-get update && \
    apt-get install -y --no-install-recommends libedgetpu1-std && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from the build stage
COPY --from=build-dev /app/alzheimer /app/alzheimer

# Copy the TFLite C library from the build stage into a standard library path
COPY --from=build-dev /usr/local/lib/libtensorflowlite_c.so /usr/lib/libtensorflowlite_c.so

# Update the dynamic linker's cache
RUN ldconfig

# The model and test images will be mounted via docker-compose.
ENTRYPOINT ["/app/alzheimer"]