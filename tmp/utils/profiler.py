from tmp.src.common_imports  import torch, time

def profile_model_once(model, input_shape=(1, 3, 640, 640)):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(3):
        model(x)

    t0 = time.time()
    model(x)
    dt = (time.time() - t0) * 1000.0
    print(f"[Profiler] One forward: {dt:.2f} ms on {device}")
