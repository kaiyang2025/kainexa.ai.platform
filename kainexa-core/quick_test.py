import torch
import transformers
from transformers import AutoTokenizer, AutoModel

print("Quick integration test...")

# 1. PyTorch GPU 테스트
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = x @ y
    print(f"✅ PyTorch GPU computation: {z.shape}")
else:
    print("⚠️  No GPU available")

# 2. Transformers 테스트
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Hello, Kainexa!"
    tokens = tokenizer(text, return_tensors="pt")
    print(f"✅ Tokenization successful: {len(tokens['input_ids'][0])} tokens")
except Exception as e:
    print(f"❌ Transformers error: {e}")

# 3. DeepSpeed 초기화 테스트
try:
    import deepspeed
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    ds_config = {
        'train_batch_size': 8,
        'fp16': {'enabled': False},
        'zero_optimization': {'stage': 0}
    }
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=ds_config
    )
    print("✅ DeepSpeed initialization successful")
except Exception as e:
    print(f"❌ DeepSpeed error: {e}")

print("\n✨ Quick test completed!")
