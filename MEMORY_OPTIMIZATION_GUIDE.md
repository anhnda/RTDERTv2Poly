# Memory Optimization Guide for RTDETRv2 Training

## Quick Fix: Use Low Memory Config

```bash
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly_lowmem.yml
```

This config is optimized for a single 16GB GPU with these changes:
- Batch size: 16 → 4
- Validation batch size: 32 → 8
- Multi-scale training: Disabled initially
- Mixed precision (AMP): Enabled

---

## Memory Usage Breakdown (16GB GPU)

**Original Config (OOM ✗)**
- Batch size: 16
- Image sizes: 480-800px (multi-scale)
- Mixed precision: OFF
- **Memory needed: ~18-20GB**

**Low Memory Config (Works ✓)**
- Batch size: 4
- Image size: 640px (fixed)
- Mixed precision: ON
- **Memory needed: ~10-12GB**

---

## Progressive Memory Optimization Strategies

### Strategy 1: Reduce Batch Size (Fastest)
```bash
# Try batch size 4 first
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly.yml \
  -u train_dataloader.total_batch_size=4 \
  val_dataloader.total_batch_size=8

# If still OOM, try batch size 2
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly.yml \
  -u train_dataloader.total_batch_size=2 \
  val_dataloader.total_batch_size=4
```

### Strategy 2: Enable Mixed Precision (Best Performance)
```bash
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly.yml \
  --use-amp \
  -u train_dataloader.total_batch_size=8
```

### Strategy 3: Disable Multi-Scale Training
```bash
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly.yml \
  -u train_dataloader.collate_fn.scales=null
```

### Strategy 4: Reduce Image Size
```bash
# Use 512x512 instead of 640x640
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly.yml \
  -u train_dataloader.dataset.transforms.ops[-4].size=[512,512]
```

### Strategy 5: Set PyTorch Memory Optimizer
```bash
# Before running training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly_lowmem.yml
```

---

## Recommended Settings by GPU Memory

| GPU Memory | Batch Size | Image Size | Multi-Scale | AMP |
|------------|------------|------------|-------------|-----|
| 8GB        | 1-2        | 512        | OFF         | ON  |
| 12GB       | 2-4        | 640        | OFF         | ON  |
| 16GB       | 4-8        | 640        | OFF/ON      | ON  |
| 24GB       | 8-16       | 640        | ON          | ON  |
| 40GB+      | 16-32      | 640-800    | ON          | Optional |

---

## Performance vs Memory Trade-offs

### Batch Size Impact
- **Smaller batch** = Less memory, more noise in gradients
- **Larger batch** = More memory, smoother training
- **Minimum recommended**: 2 (below this, training may be unstable)

### Mixed Precision (AMP) Impact
- **Memory savings**: ~30-40%
- **Speed**: ~2x faster
- **Accuracy**: Minimal impact (0-0.5% mAP difference)
- **Recommendation**: Always enable for RTDETRv2

### Multi-Scale Training Impact
- **Memory**: +20-50% depending on scale range
- **mAP improvement**: +1-2%
- **Recommendation**: Disable initially, enable after confirming stable training

---

## Troubleshooting

### Error: "CUDA out of memory"
1. Reduce batch size by half
2. Enable --use-amp if not already
3. Disable multi-scale training
4. Set PYTORCH_CUDA_ALLOC_CONF

### Error: "Loss is NaN"
- If batch size is 1 and using BatchNorm, enable sync_bn=False
- Check if learning rate is too high with small batch size
- Consider accumulating gradients (not natively supported, need code modification)

### Slow Training
- Increase num_workers (default: 4, try: 8)
- Enable mixed precision (--use-amp)
- Check if data loading is bottleneck with nvidia-smi

---

## Monitoring GPU Usage

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Or use
gpustat -i 1
```

Look for:
- **Memory usage stabilizing** after first few iterations (good sign)
- **Memory increasing indefinitely** (memory leak, stop training)
- **GPU utilization > 80%** (good, compute-bound)
- **GPU utilization < 50%** (data loading bottleneck, increase num_workers)
