# Convolutional Neural Networks (CNN) — Interview Questions & Answers

---

## Small Dataset Example (Image Classification)

```
3×3 grayscale image patch (pixel values 0-255):

Image 1 (Horizontal edge):
[255, 255, 255]
[  0,   0,   0]
[  0,   0,   0]

Image 2 (Vertical edge):
[255,   0,   0]
[255,   0,   0]
[255,   0,   0]

Image 3 (Uniform):
[128, 128, 128]
[128, 128, 128]
[128, 128, 128]

Horizontal edge detector kernel (3×3):
[ 1,  1,  1]
[ 0,  0,  0]
[-1, -1, -1]
```

### Dataset Questions

**Q: Apply the horizontal edge kernel to Image 1 (valid convolution). What is the output value at the center?**
> **A:** Center output = (255×1 + 255×1 + 255×1) + (0×0 + 0×0 + 0×0) + (0×(-1) + 0×(-1) + 0×(-1)) = 765 + 0 − 0 = **765** (strong horizontal edge detected).

**Q: Apply the same kernel to Image 3 (uniform). What is the output?**
> **A:** (128+128+128) + 0 − (128+128+128) = 384 − 384 = **0**. No edge detected in a uniform region — exactly correct behavior.

**Q: If the input is 28×28, filter is 5×5, stride=1, padding=0, how many filters needed, and what is output size?**
> **A:** Output size = (28 − 5 + 0×2) / 1 + 1 = **24×24**. Number of filters is a hyperparameter (e.g., 32 filters → output is 24×24×32). Each filter detects a different pattern.

---

## Questions & Answers

---

### Q1. What is a CNN?

**A:** A Convolutional Neural Network (CNN) is a deep learning architecture designed for grid-like data (images, audio spectrograms). Instead of fully connected layers, CNNs use:
- **Convolutional layers:** Learn spatial filters/features
- **Pooling layers:** Reduce spatial dimensions
- **Fully connected layers:** Final classification/regression

Key advantages: **parameter sharing** (same filter applied everywhere) and **local connectivity** (neurons only connect to a local patch) → far fewer parameters than fully connected.

---

### Q2. What are convolution layers?

**A:** A convolutional layer slides a learnable filter (kernel) across the input to produce a feature map:

```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Filter[m,n] + bias
```

Each filter detects a specific pattern (edge, corner, texture). Multiple filters → multiple feature maps.

**Parameters:** kernel_size, n_filters (out_channels), stride, padding

---

### Q3. What is a filter/kernel in CNN?

**A:** A filter is a small learnable weight matrix (e.g., 3×3, 5×5) that detects specific local patterns:
- Early layers: edges, colors, simple textures
- Middle layers: shapes, object parts
- Deep layers: complex object features (eyes, wheels, etc.)

A CNN with 32 filters of size 3×3×3 (RGB) has 32 × (3×3×3 + 1) = 896 parameters in that layer.

---

### Q4. What is stride?

**A:** Stride is the number of pixels the filter moves at each step:
- **Stride=1:** Filter moves 1 pixel at a time → larger output
- **Stride=2:** Filter jumps 2 pixels → smaller output (similar to pooling)

```
Output size = (Input − Filter) / Stride + 1
```
Higher stride = fewer output positions = downsampling.

---

### Q5. What is padding?

**A:** Padding adds zeros around the input border:
- **Valid (no padding):** Output shrinks — `out = (in − kernel + 1)`
- **Same padding:** Output same size as input — adds `kernel//2` zeros on each side

```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # same padding
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)  # valid
```

Padding prevents boundary information loss and makes output size predictable.

---

### Q6. What is a feature map?

**A:** A feature map is the output of applying one filter to the input. It represents the presence/strength of the detected pattern at each spatial location.

- One filter → one 2D feature map
- N filters → N feature maps stacked → output shape: (N, H_out, W_out)

Early feature maps detect simple patterns; deeper ones detect complex structures.

---

### Q7. What is max pooling?

**A:** Max pooling takes the maximum value in each pooling window, reducing spatial dimensions:

```
Input patch: [[2, 4], [3, 1]] → Max = 4
```

```python
nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W
```

Benefits: Reduces computation, provides translation invariance (small shifts in input don't change output), retains most prominent features.

---

### Q8. What is average pooling?

**A:** Average pooling takes the mean value in each pooling window — smoother than max pooling:

```python
nn.AvgPool2d(kernel_size=2, stride=2)
```

Less aggressive than max pooling. Used in some architectures (Inception). Global average pooling (GAP) averages across the entire spatial dimension → used instead of flatten+FC in modern architectures.

---

### Q9. What is global average pooling?

**A:** Global Average Pooling (GAP) computes the mean of each entire feature map:
- Input: (batch, C, H, W) → Output: (batch, C)
- Replaces the flatten + fully connected layer combination
- Drastically reduces parameters
- Provides spatial invariance
- Used in ResNet, MobileNet, EfficientNet

```python
nn.AdaptiveAvgPool2d(1)  # reduces H,W to 1×1
x = x.squeeze(-1).squeeze(-1)  # → (batch, C)
```

---

### Q10. What is a fully connected layer in CNN?

**A:** After convolutional and pooling layers, the feature maps are flattened into a 1D vector and passed through fully connected (dense) layers for final classification/regression.

Modern CNNs often replace FC layers with GAP to reduce overfitting and parameters. FC layers are still used in the final 1–2 layers before output.

---

### Q11. What is the output size formula for a conv layer?

**A:**
```
H_out = floor((H_in + 2P − K) / S) + 1
W_out = floor((W_in + 2P − K) / S) + 1
```
- `H_in/W_in` = input height/width
- `P` = padding
- `K` = kernel size
- `S` = stride

Example: Input 32×32, K=3, P=1, S=1 → (32 + 2 − 3)/1 + 1 = **32×32** (same padding)

---

### Q12. What is the difference between valid and same padding?

**A:**
| | Valid (no padding) | Same padding |
|---|---|---|
| Output size | Smaller: `(in − K + 1)` | Same: `in / S` |
| Border handling | Ignores borders | Pads with zeros |
| Info loss | Loses border info | Preserves all spatial info |
| Use case | When shrinkage is OK | When maintaining size matters |

---

### Q13. What is transfer learning in CNNs?

**A:** Use a pretrained CNN (trained on ImageNet with millions of images) for a new task:

```python
import torchvision.models as models

model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(2048, n_classes)  # Only this layer trains
```

Early layers learn generic features (edges, textures) that transfer across domains. Only fine-tune later layers for domain-specific features.

---

### Q14. What is data augmentation in image tasks?

**A:** Data augmentation applies random transformations during training to increase effective dataset size and improve generalization:

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomErasing(p=0.2),
])
```

Key: augmentations should be class-preserving (a flipped cat is still a cat).

---

### Q15. What is ResNet and what are skip connections?

**A:** ResNet (Residual Network) introduced skip connections that bypass one or more layers:
```
F(x) + x  (instead of just F(x))
```

**Why it works:**
- Gradient flows directly through skip connection → no vanishing gradient
- Easier to learn identity (F(x)=0) than arbitrary transformations
- Enables training of very deep networks (ResNet-152 has 152 layers)

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(out + identity)  # skip connection
```

---

### Q16. What is object detection?

**A:** Object detection finds and classifies multiple objects in an image — outputs bounding boxes + class labels + confidence scores.

**Evolution:**
- **Two-stage:** R-CNN → Fast R-CNN → Faster R-CNN (region proposals + classification)
- **One-stage:** YOLO, SSD, RetinaNet (single forward pass → faster, slightly less accurate)
- **Transformer-based:** DETR (end-to-end, no NMS needed)

---

### Q17. What is image segmentation?

**A:** Image segmentation classifies every pixel in an image:
- **Semantic segmentation:** Each pixel → class label (all cats same label)
- **Instance segmentation:** Each pixel → specific object instance (cat1 vs cat2)
- **Panoptic segmentation:** Combines both

Architectures: U-Net (medical), DeepLab, Mask R-CNN (instance), SegFormer.

---

### Q18. What is depthwise separable convolution?

**A:** Depthwise separable convolution factorizes a standard conv into two steps:
1. **Depthwise:** Apply one filter per input channel independently
2. **Pointwise:** 1×1 conv to combine across channels

```
Standard 3×3 conv: K × K × C_in × C_out
Depthwise sep:     K × K × C_in   +   C_in × C_out
Speedup:           ~8-9x fewer operations
```

Used in MobileNet, EfficientNet for fast inference on mobile devices.

---

### Q19. What is the inductive bias of CNNs?

**A:** CNNs have two key inductive biases:
1. **Translation equivariance:** Applying a filter to a shifted input gives a shifted output — the same feature detector works everywhere
2. **Local connectivity:** Neurons only look at a small local patch — assumes nearby pixels are related

These biases are useful for images (nearby pixels are related, features appear at multiple locations) but may not hold for all data types (e.g., tabular data).

---

### Q20. What is the difference between classification, detection, and segmentation?

**A:**
| Task | Output | Example |
|---|---|---|
| Classification | Single class label | "This image is a cat" |
| Detection | Bounding boxes + classes | "Cat at (x1,y1,x2,y2), Dog at ..." |
| Semantic Segmentation | Per-pixel class | Each pixel labeled: cat/dog/background |
| Instance Segmentation | Per-pixel + instance ID | Cat 1 pixels, Cat 2 pixels separate |
| Panoptic Segmentation | Semantic + Instance | Full scene understanding |

---

## Quick Code Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 → 28x28
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)                               # 28x28 → 14x14

        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 → 14x14
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)                               # 14x14 → 7x7

        # Classifier
        self.gap    = nn.AdaptiveAvgPool2d(1)                      # 7x7 → 1x1
        self.fc     = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

model = SimpleCNN(n_classes=10)
x = torch.randn(4, 1, 28, 28)  # batch of 4 grayscale 28x28 images
out = model(x)
print("Output shape:", out.shape)  # (4, 10)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
```
