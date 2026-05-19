"""Test script to validate handling of empty images (no annotations)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.zoo.rtdetr.matcher import HungarianMatcher
from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2

print("Testing empty image handling...")

# Test 1: Matcher with empty targets
print("\n1. Testing HungarianMatcher with empty targets...")
matcher = HungarianMatcher(
    weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
    use_focal_loss=True
)

# Create fake outputs
batch_size = 4
num_queries = 300
num_classes = 1
outputs = {
    'pred_logits': torch.randn(batch_size, num_queries, num_classes),
    'pred_boxes': torch.rand(batch_size, num_queries, 4)
}

# Test case 1a: All empty targets
print("   - Testing all empty targets...")
empty_targets = [
    {'labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.zeros(0, 4)},
    {'labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.zeros(0, 4)},
    {'labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.zeros(0, 4)},
    {'labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.zeros(0, 4)},
]

try:
    result = matcher(outputs, empty_targets)
    print(f"   ✓ Matcher handled all empty targets. Indices: {len(result['indices'])} batches")
    for i, (src, tgt) in enumerate(result['indices']):
        assert len(src) == 0 and len(tgt) == 0, f"Expected empty indices, got src={len(src)}, tgt={len(tgt)}"
    print("   ✓ All indices are correctly empty")
except Exception as e:
    print(f"   ✗ Matcher failed with all empty targets: {e}")
    sys.exit(1)

# Test case 1b: Mixed empty and non-empty targets
print("   - Testing mixed empty and non-empty targets...")
mixed_targets = [
    {'labels': torch.tensor([0], dtype=torch.int64), 'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]])},
    {'labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.zeros(0, 4)},
    {'labels': torch.tensor([0, 0], dtype=torch.int64), 'boxes': torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])},
    {'labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.zeros(0, 4)},
]

try:
    result = matcher(outputs, mixed_targets)
    print(f"   ✓ Matcher handled mixed targets. Indices lengths: {[len(src) for src, _ in result['indices']]}")
    assert len(result['indices'][0][0]) > 0, "Expected non-empty indices for image 0"
    assert len(result['indices'][1][0]) == 0, "Expected empty indices for image 1"
    assert len(result['indices'][2][0]) > 0, "Expected non-empty indices for image 2"
    assert len(result['indices'][3][0]) == 0, "Expected empty indices for image 3"
    print("   ✓ Indices correctly match target presence")
except Exception as e:
    print(f"   ✗ Matcher failed with mixed targets: {e}")
    sys.exit(1)

# Test 2: Criterion with empty targets
print("\n2. Testing RTDETRCriterionv2 with empty targets...")
criterion = RTDETRCriterionv2(
    matcher=matcher,
    weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
    losses=['vfl', 'boxes'],
    num_classes=1
)

# Test case 2a: All empty targets
print("   - Testing criterion with all empty targets...")
try:
    losses = criterion(outputs, empty_targets)
    print(f"   ✓ Criterion handled all empty targets. Losses: {losses}")
    for key, value in losses.items():
        assert torch.isfinite(value), f"Loss {key} is not finite: {value}"
        assert value.item() >= 0, f"Loss {key} is negative: {value}"
    print("   ✓ All losses are finite and non-negative")
except Exception as e:
    print(f"   ✗ Criterion failed with all empty targets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test case 2b: Mixed targets
print("   - Testing criterion with mixed targets...")
try:
    losses = criterion(outputs, mixed_targets)
    print(f"   ✓ Criterion handled mixed targets. Losses: {losses}")
    for key, value in losses.items():
        assert torch.isfinite(value), f"Loss {key} is not finite: {value}"
    print("   ✓ All losses are finite")
except Exception as e:
    print(f"   ✗ Criterion failed with mixed targets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Dataset loading
print("\n3. Testing CocoDetection with wli_train.json...")

# Check if annotation file exists
ann_file = '/Users/anhnd/CodingSpace/RT-DETR/rtdetrv2_pytorch/wli_train.json'
if not os.path.exists(ann_file):
    print(f"   ⚠ Annotation file not found, skipping dataset test")
else:
    try:
        from src.data.dataset.coco_dataset import CocoDetection

        # Try to find the image folder
        img_folder = None
        possible_folders = [
            '/Users/anhnd/CodingSpace/RT-DETR/rtdetrv2_pytorch/wli_train_images/',
            '/data/poly/images/',
            'wli_train_images/',
            'images/'
        ]
        for folder in possible_folders:
            if os.path.exists(folder):
                img_folder = folder
                break

        if img_folder is None:
            print(f"   ⚠ Image folder not found, testing with category2label only...")
            # Just test that we can load the annotation file and check categories
            import json
            with open(ann_file) as f:
                data = json.load(f)
            print(f"   ✓ Annotation file loaded: {len(data['images'])} images")
            print(f"   - Categories: {data['categories']}")
            print(f"   - Expected category2label mapping: {{1: 0}}")
        else:
            dataset = CocoDetection(
                img_folder=img_folder,
                ann_file=ann_file,
                transforms=None,
                return_masks=False,
                remap_mscoco_category=False
            )

            print(f"   ✓ Dataset loaded: {len(dataset)} images")

            # Check category2label mapping
            print(f"   - category2label: {dataset.category2label}")
            print(f"   - Categories: {dataset.categories}")

            # Find an image without annotations (from our earlier analysis, image IDs 2048+ have no annotations)
            empty_img_id = 2048
            if empty_img_id < len(dataset):
                print(f"   - Testing image without annotations (id={empty_img_id})...")
                image, target = dataset.load_item(empty_img_id)
                print(f"     Labels shape: {target['labels'].shape}, Boxes shape: {target['boxes'].shape}")
                assert target['labels'].shape[0] == 0, "Expected empty labels"
                assert target['boxes'].shape[0] == 0, "Expected empty boxes"
                print(f"   ✓ Empty image loaded correctly")

            # Test an image with annotations
            print(f"   - Testing image with annotations (id=0)...")
            image, target = dataset.load_item(0)
            print(f"     Labels: {target['labels']}, Boxes shape: {target['boxes'].shape}")
            assert target['labels'].shape[0] > 0, "Expected non-empty labels"
            # Check that labels are remapped to 0
            assert torch.all(target['labels'] == 0), f"Expected labels to be 0, got {target['labels']}"
            print(f"   ✓ Non-empty image loaded correctly with remapped labels")

    except Exception as e:
        print(f"   ✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! The code can handle empty images.")
print("="*60)
