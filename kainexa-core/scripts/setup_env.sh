# scripts/setup_env.sh
#!/bin/bash
set -e

echo "Setting up Kainexa Core development environment..."

# Python 버전 확인
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1,2)
echo "Python version: $PYTHON_VERSION"

# 가상 환경 생성
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# 가상 환경 활성화
source venv/bin/activate

# pip 업그레이드
echo "Upgrading pip..."
pip install --upgrade pip

# 의존성 설치
echo "Installing dependencies..."
pip install -r requirements.txt

# 개발 의존성 설치 (있을 경우)
if [ -f "requirements-dev.txt" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"