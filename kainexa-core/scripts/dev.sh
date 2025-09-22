# scripts/dev.sh
#!/bin/bash
set -e

echo "Starting Kainexa Core development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Start services
docker-compose up -d postgres redis qdrant

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 5

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting API server..."
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000