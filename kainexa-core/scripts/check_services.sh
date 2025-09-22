# scripts/check_services.sh
#!/bin/bash

echo "Checking service status..."

# PostgreSQL
if docker-compose exec -T postgres pg_isready -U kainexa > /dev/null 2>&1; then
    echo "✅ PostgreSQL: Running"
else
    echo "❌ PostgreSQL: Not running"
fi

# Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Running"
else
    echo "❌ Redis: Not running"
fi

# Qdrant
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant: Running"
else
    echo "❌ Qdrant: Not running"
fi

# API
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ API: Running"
else
    echo "❌ API: Not running"
fi