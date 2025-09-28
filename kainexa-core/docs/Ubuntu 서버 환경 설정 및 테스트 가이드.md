Kainexa AI Agent Platform - Ubuntu 서버 환경 설정 및 테스트 가이드
📋 시스템 요구사항 확인

OS: Ubuntu 24.04.3 LTS
GPU: NVIDIA RTX 3090 ×2 (24GB ×2)
RAM: 최소 32GB 권장
Storage: 최소 100GB 여유 공간


1️⃣ 기본 시스템 설정
Step 1.1: 시스템 업데이트
bash# 시스템 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 도구 설치
sudo apt install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    net-tools \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release
Step 1.2: Python 3.11 설치
bash# Python 3.11 설치
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# 기본 Python 설정
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --config python3

# pip 업그레이드
python3 -m pip install --upgrade pip

2️⃣ NVIDIA GPU 드라이버 및 CUDA 설정
Step 2.1: NVIDIA 드라이버 설치
bash# 기존 드라이버 제거
sudo apt-get purge nvidia* -y
sudo apt-get autoremove -y

# NVIDIA 드라이버 설치 (535 버전 - RTX 3090 지원)
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install -y nvidia-driver-535

# 재부팅
sudo reboot
Step 2.2: CUDA 11.8 설치
bash# 재부팅 후 드라이버 확인
nvidia-smi

# CUDA 11.8 설치
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# @ CUDA 11.8 설치 오류 시  2가지 방법 #####################################################
# Ubuntu 24.04는 GCC 13.3을 사용하는데, CUDA 11.8은 GCC 11까지만 공식 지원
# 1

# 1. CUDA 12.3 설치 다운로드 (GCC 13 지원)
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run

# 2. 설치
sudo sh cuda_12.3.2_545.23.08_linux.run --toolkit --silent

# 3. 환경 변수 설정
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. PyTorch는 CUDA 12.1 지원
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121


# 2

# GCC 11 설치 및 CUDA 11.8 설치
# 1. GCC 11 설치
sudo apt install gcc-11 g++-11 -y

# 2. CUDA 설치 시 GCC 11 명시적 지정
sudo sh cuda_11.8.0_520.61.05_linux.run \
    --toolkit \
    --override \
    --silent \
    --toolkitpath=/usr/local/cuda-11.8

# 3. 설치 확인
ls /usr/local/cuda-11.8/

# 4. 환경 변수 설정
cat >> ~/.bashrc << 'EOF'
# CUDA 11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
EOF

source ~/.bashrc

# 5. CUDA 설치 확인
nvcc --version
nvidia-smi

# ########################################################

# CUDA 설치 확인
nvcc --version

Step 2.3: cuDNN 설치
bash# cuDNN 8.6 다운로드 (NVIDIA 계정 필요)
# https://developer.nvidia.com/cudnn 에서 다운로드 후

tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*


Step 2.3: cuDNN 설치 2
# https://developer.nvidia.com/cudnn

wget https://developer.download.nvidia.com/compute/cudnn/9.13.0/local_installers/cudnn-local-repo-ubuntu2404-9.13.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.13.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.13.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn

To install for CUDA 12, perform the above configuration but install the CUDA 12 specific package:
sudo apt-get -y install cudnn9-cuda-12

Step 2.4: NVLink 설정 확인
bash# NVLink 상태 확인
nvidia-smi nvlink -s
nvidia-smi topo -m

# P2P 통신 활성화
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1695  # RTX 3090 최대 클럭

# 전력 제한 설정 (350W - RTX 3090 TDP)
sudo nvidia-smi -pl 350

# 현재 설정 확인
nvidia-smi -q -d PERFORMANCE

3️⃣ Docker 및 Docker Compose 설치
Step 3.1: Docker 설치
bash# Docker GPG 키 추가
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Docker 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER
newgrp docker

# Docker 서비스 시작
sudo systemctl enable docker
sudo systemctl start docker

Step 3.2: NVIDIA Container Toolkit 설치

bash# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# GPU 지원 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

4️⃣ Kainexa 프로젝트 설정
Step 4.1: 프로젝트 클론 및 설정
bash# 프로젝트 디렉토리 생성
mkdir -p ~/kainexa
cd ~/kainexa

# Git 클론 (실제 저장소 URL로 변경)
git clone https://github.com/kaiyang2025/kainexa.ai.platform.git
cd kainexa-core

# 환경 설정 파일 생성
cp .env.example .env

# .env 파일 수정
vim .env
Step 4.2: .env 파일 설정
bash# .env 파일 내용
APP_NAME="Kainexa Core"
APP_VERSION="0.1.0"
ENVIRONMENT="production"
DEBUG=False

# API
API_PREFIX="/api/v1"
ALLOWED_ORIGINS="http://localhost:3000,http://your-domain.com"

# Database
DATABASE_URL="postgresql+asyncpg://kainexa:your_password@localhost:5432/kainexa_db"
REDIS_URL="redis://localhost:6379"

# Qdrant
QDRANT_HOST="localhost"
QDRANT_PORT=6333

# GPU 설정
CUDA_VISIBLE_DEVICES="0,1"
TENSOR_PARALLEL_SIZE=2

# Security
SECRET_KEY="change-this-to-secure-key-$(openssl rand -hex 32)"
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Model 경로
MODEL_PATH="/models"
SOLAR_MODEL="upstage/solar-10.7b-instruct"


Step 4.3: Python 가상환경 설정
bash# 가상환경 생성
python3.11 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# GPU 의존성 설치
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt

5️⃣ 데이터베이스 및 서비스 시작
Step 5.1: Docker 서비스 시작
bash# Docker 서비스 시작 (PostgreSQL, Redis, Qdrant)
make docker-up

# 또는 직접 실행
docker compose up -d postgres redis qdrant

# 서비스 상태 확인
docker ps
docker compose ps

Step 5.2: 데이터베이스 초기화
bash# 데이터베이스 마이그레이션
source venv/bin/activate
alembic upgrade head

# 또는 Makefile 사용
make migrate

Step 5.3: 모니터링 스택 시작
bash# Prometheus, Grafana, Jaeger 시작
make monitor-up

# 또는 직접 실행
docker-compose -f docker-compose.monitoring.yml up -d

# 접속 확인
# Grafana: http://localhost:3000 (admin/kainexa123)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686

6️⃣ 모델 다운로드 및 설정
Step 6.1: Solar 모델 다운로드
bash# 모델 디렉토리 생성
sudo mkdir -p /models
sudo chown $USER:$USER /models

# Hugging Face CLI 설치
pip install huggingface-hub

# 모델 다운로드 (약 20GB)
python3 << EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="upstage/solar-10.7b-instruct",
    local_dir="/models/solar-10.7b",
    local_dir_use_symlinks=False
)
EOF
Step 6.2: 모델 로드 테스트
bash# GPU 메모리 및 모델 로드 테스트
python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

# 모델 로드 테스트 (메모리 체크)
model_path = "/models/solar-10.7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully!")
EOF
# ############################
# 1. 환경 준비
cd ~/kainexa.ai.platform/kainexa-core
source venv/bin/activate

# 2. Docker 서비스 시작
docker compose up -d

# 3. 데이터베이스 초기화
python scripts/init_database.py

# 4. Solar 모델 다운로드 (처음 한 번만)
python scripts/download_solar_model.py --download

# 5. 샘플 문서 업로드
python scripts/upload_documents.py

# 6. 통합 API 서버 시작
python src/api/main_integrated.py

# 7. 새 터미널에서 테스트 실행
python scripts/test_api.py

# 8. 개별 시나리오 테스트
python src/scenarios/production_monitoring.py
python src/scenarios/predictive_maintenance.py
python src/scenarios/quality_control.py

# ##############################

7️⃣ 애플리케이션 시작 및 테스트
Step 7.1: 개발 모드 실행
bash# 개발 서버 시작
make dev

# 또는 직접 실행
source venv/bin/activate
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
Step 7.2: GPU 병렬 모드 실행
bash# 텐서 병렬 실행 (2 GPU)
make gpu-run

# 또는 직접 실행
./scripts/run_gpu_parallel.sh parallel
Step 7.3: 프로덕션 모드 실행
bash# 전체 스택 시작
make full-stack

# 또는 개별 실행
# 1. 서비스 시작
docker-compose up -d

# 2. API 서버 시작 (Gunicorn)
gunicorn src.api.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log

8️⃣ 테스트 실행
Step 8.1: 헬스체크
bash# API 헬스체크
curl http://localhost:8000/api/v1/health

# 상세 헬스체크
curl http://localhost:8000/api/v1/health/detailed
Step 8.2: 통합 테스트
bash# 테스트 실행
source venv/bin/activate
pytest tests/ -v

# 특정 테스트만 실행
pytest tests/test_integration.py -v
Step 8.3: 제조업 시나리오 데모
bash# 시나리오 실행
python src/scenarios/manufacturing_demo.py

# 또는 Makefile 사용
make demo
Step 8.4: API 테스트 (Postman/curl)
bash# 1. 인증 토큰 획득
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# 2. 채팅 API 테스트
TOKEN="your-token-here"
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"지난달 매출 현황 알려줘","session_id":"test-001"}'

# 3. WebSocket 테스트
wscat -c ws://localhost:8000/api/v1/conversations/test-001/ws \
  -H "Authorization: Bearer $TOKEN"
Step 8.5: 부하 테스트
bash# Locust 설치
pip install locust

# 부하 테스트 실행
locust -f tests/load_test.py --host=http://localhost:8000
# 브라우저에서 http://localhost:8089 접속

9️⃣ 모니터링 및 로그 확인
Step 9.1: 시스템 모니터링
bash# GPU 모니터링
watch -n 1 nvidia-smi

# 시스템 리소스 모니터링
htop

# Docker 컨테이너 모니터링
docker stats
Step 9.2: 로그 확인
bash# API 로그
tail -f logs/app_*.log

# Docker 로그
docker-compose logs -f api
docker-compose logs -f postgres

# 시스템 로그
journalctl -f -u docker
Step 9.3: 메트릭 대시보드
bash# Grafana 접속
firefox http://localhost:3000
# Login: admin / kainexa123

# Prometheus 메트릭 확인
firefox http://localhost:9090

# Jaeger 트레이싱
firefox http://localhost:16686

🔟 문제 해결 (Troubleshooting)
GPU 메모리 부족
bash# GPU 메모리 정리
sudo nvidia-smi --gpu-reset

# 프로세스 종료
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>
Docker 권한 문제
bash# Docker 소켓 권한
sudo chmod 666 /var/run/docker.sock

# Docker 재시작
sudo systemctl restart docker
모델 로딩 실패
bash# 캐시 정리
rm -rf ~/.cache/huggingface/

# 모델 재다운로드
python -c "from transformers import AutoModel; AutoModel.from_pretrained('upstage/solar-10.7b-instruct', force_download=True)"

✅ 최종 확인 체크리스트
bash# 시스템 상태 종합 확인
make status

# 또는 수동 확인
echo "=== System Check ==="
echo "1. GPU Status:"
nvidia-smi
echo ""
echo "2. Docker Services:"
docker-compose ps
echo ""
echo "3. API Health:"
curl -s http://localhost:8000/api/v1/health | jq
echo ""
echo "4. Database Connection:"
docker exec kainexa-postgres pg_isready -U kainexa
echo ""
echo "5. Redis Connection:"
docker exec kainexa-redis redis-cli ping
echo ""
echo "6. Qdrant Status:"
curl -s http://localhost:6333/health
📊 성능 벤치마크
bash# GPU 벤치마크
make gpu-benchmark

# 추론 성능 테스트
python3 tests/benchmark/inference_benchmark.py

# 예상 결과 (RTX 3090 x2, Solar-10.7B)
# - 단일 GPU: ~15-20 tokens/sec
# - 텐서 병렬: ~25-35 tokens/sec
# - 첫 토큰 지연: ~2-3초