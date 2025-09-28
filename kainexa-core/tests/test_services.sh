#!/bin/bash

echo "================================"
echo "Kainexa Services Status"
echo "================================"
echo ""

# 서버 IP
SERVER_IP=$(hostname -I | awk '{print $1}')
echo "Server IP: $SERVER_IP"
echo ""

# 컨테이너 상태
echo "Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep kainexa
echo ""

# 서비스 테스트
echo "Service Tests:"
echo -n "  PostgreSQL (5432): "
docker exec kainexa-postgres pg_isready -U kainexa &>/dev/null && echo "✅ Ready" || echo "❌ Not ready"

echo -n "  Redis (6379):      "
docker exec kainexa-redis redis-cli ping 2>/dev/null | grep -q PONG && echo "✅ Ready" || echo "❌ Not ready"

echo -n "  Qdrant (6333):     "
curl -s http://localhost:6333/ &>/dev/null && echo "✅ Ready" || echo "❌ Not ready"

echo ""
echo "================================"
echo "External Access URLs:"
echo "================================"
echo "  PostgreSQL: postgresql://kainexa:password@$SERVER_IP:5432/kainexa_db"
echo "  Redis:      redis://$SERVER_IP:6379"
echo "  Qdrant:     http://$SERVER_IP:6333/dashboard"
echo "  API:        http://$SERVER_IP:8000/api/v1/docs (when running)"
echo "================================"
