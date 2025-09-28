# scripts/run.sh
#!/bin/bash

MODE=${1:-"dev"}
WORKERS=${2:-"1"}

case $MODE in
    dev)
        echo "Starting development server..."
        uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
        ;;
    
    gpu)
        echo "Starting GPU inference mode..."
        CUDA_VISIBLE_DEVICES=0,1 python src/models/inference.py
        ;;
    
    prod)
        echo "Starting production server..."
        gunicorn src.api.main:app \
            -w $WORKERS \
            -k uvicorn.workers.UvicornWorker \
            --bind 0.0.0.0:8000
        ;;
    
    demo)
        echo "Running manufacturing demo..."
        python src/scenarios/manufacturing_demo.py
        ;;
    
    *)
        echo "Usage: $0 {dev|gpu|prod|demo} [workers]"
        exit 1
        ;;
esac