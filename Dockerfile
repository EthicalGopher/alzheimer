
FROM golang:1.24.5-alpine AS builder


WORKDIR /src

COPY go.mod ./
RUN go mod download


COPY . .

RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /app/main .


# --- Stage 2: Create the final production image ---

FROM python:3.11-slim


WORKDIR /app



# This Dockerfile expects the requirements.txt file to be at 'python/requirements.txt'
COPY python/requirements.txt .

# Install the Python dependencies. --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app/main .

COPY python/ ./python

COPY public/ ./public

COPY alzheimers_model.onnx .

EXPOSE 3000

CMD ["./main"]