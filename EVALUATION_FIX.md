# Evaluation Fix: Zero AP/AR Issue Resolved

## Problem

After training started, all validation metrics showed 0.000:
```
Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 ] = 0.000
```

## Root Cause

The issue was a **category ID mismatch** between model predictions and COCO evaluation:

1. **Dataset**: Polyps have `category_id=1` in wli_train.json
2. **Training**: We remap `category_id=1 → label=0` for the model (correct)
3. **Prediction**: Model outputs `label=0` (correct)
4. **Evaluation**: COCO evaluator expects `category_id=1` (mismatch!)

The postprocessor was outputting `labels=0` but the evaluator needed `category_id=1` to match the ground truth annotations.

## Solution

Modified three files to add automatic label-to-category remapping:

### 1. `src/zoo/rtdetr/rtdetr_postprocessor.py`

Added `label2category` mapping support:

```python
class RTDETRPostProcessor(nn.Module):
    def __init__(self, ...):
        # Added
        self.label2category = None

    def set_label2category(self, label2category: dict):
        """Set label to category ID mapping for non-MSCOCO datasets"""
        self.label2category = label2category

    def forward(self, outputs, orig_target_sizes):
        # ... model predictions ...

        # Added: Remap labels to category IDs
        if self.label2category is not None:
            labels = torch.tensor([self.label2category[int(x.item())]
                                  for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)
```

**What this does**: Converts model predictions (`label=0`) back to COCO category IDs (`category_id=1`)

### 2. `src/solver/_solver.py`

Added automatic setup of label2category mapping from the validation dataset:

```python
# Set label2category mapping for postprocessor from validation dataset
if hasattr(self.postprocessor, 'set_label2category'):
    val_dataset = self.val_dataloader.dataset
    # Unwrap DistributedDataLoader if needed
    if hasattr(val_dataset, 'dataset'):
        val_dataset = val_dataset.dataset
    if hasattr(val_dataset, 'label2category'):
        label2category = val_dataset.label2category
        self.postprocessor.set_label2category(label2category)
        print(f'Set postprocessor label2category mapping: {label2category}')
```

**What this does**: Automatically extracts `{0: 1}` mapping from the dataset and configures the postprocessor

### 3. `src/data/dataset/coco_dataset.py` (already fixed)

Enabled automatic category remapping:

```python
def load_item(self, idx):
    # ...
    # Changed from: image, target = self.prepare(image, target)
    # To:
    image, target = self.prepare(image, target, category2label=self.category2label)
```

**What this does**: Maps `category_id=1 → label=0` during training

## How It Works

### Training Flow:
```
Annotation (category_id=1)
  → Dataset remaps to (label=0)
  → Model trains with (label=0)
```

### Evaluation Flow:
```
Model predicts (label=0)
  → Postprocessor remaps to (category_id=1)
  → COCO evaluator matches with GT (category_id=1)
  → ✓ Correct evaluation!
```

## Expected Output

When training restarts, you should see:

```bash
Set postprocessor label2category mapping: {0: 1}
```

This confirms the fix is active. After the first evaluation epoch, you should see non-zero metrics:

```
Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.XXX  (non-zero!)
Average Recall     (AR) @[ IoU=0.50:0.95 ] = 0.XXX  (non-zero!)
```

## Verification

To verify the fix is working:

1. **Check console output**: Look for "Set postprocessor label2category mapping: {0: 1}"
2. **Monitor validation metrics**: After epoch 5-10, AP should be > 0.0
3. **Check tensorboard**: Validation metrics should show non-zero values

## Why This Matters

- ✅ Model can now properly evaluate on single-class datasets
- ✅ Works with any COCO-format dataset (not just MSCOCO)
- ✅ Automatic - no manual config needed
- ✅ Backward compatible with MSCOCO (via `remap_mscoco_category` flag)

## Notes

- This fix only affects **evaluation**, not training
- Training losses (loss_vfl, loss_bbox, loss_giou) were always correct
- The fix applies to both `train()` and `eval()` modes
- Works with distributed training (automatically unwraps dataloaders)
