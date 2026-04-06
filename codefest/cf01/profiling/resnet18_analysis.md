# ResNet-18 Profiling Analysis

**Model**: ResNet-18  
**Input**: (1, 3, 224, 224) FP32 
**Tool**: torchinfo
**Total Mult-Adds**: 1.81 GMACs 
**Total Parameters**: 11,689,512

---
##Top 5 MAC-Intensive Layers

| Rank |  Layer Name  |    Input Shape   |   Output Shape    | Parameters |  Mult-Adds  |
| :----- | :------ | :------ | :--- | :----- | :------------- |
|  1  | Conv2d: 1-1  | [1, 3, 224, 224] | [1, 64, 112, 112] |   9,408    | 118,013,952 |
|  2  | Conv2d: 3-1  | [1, 64, 56, 56]  | [1, 64, 56, 56]   |   36,864   | 115,605,504 |
|  3  | Conv2d: 3-4  | [1, 64, 56, 56]  | [1, 64, 56, 56]   |   36,864   | 115,605,504 |
|  4  | Conv2d: 3-7  | [1, 64, 56, 56]  | [1, 64, 56, 56]   |   36,864   | 115,605,504 |
|  5  | Conv2d: 3-10 | [1, 64, 56, 56]  | [1, 64, 56, 56]   |   36,864   | 115,605,504 |

Note: Many ResNet-18 Conv2d layers at 56×56 spatial resolution share exactly 115,605,504 MACs. The stem conv1 (7×7, stride 2) is the single highest due to the large 224×224 input spatial map, despite having the fewest parameters.

## Arthmetic Intensity - Most MAC-Intensive Layer

Step 1: FLOPs
```
FLOPs = 2 × MACs = 2 × 118,013,952 = 236,027,904 FLOPs
```

Step 2: Weight Bytes (loaded from DRAM, no reuse)
```
Params       = 3 × 64 × 7 × 7 = 9,408 weights
Weight bytes = 9,408 × 4 bytes (FP32) = 37,632 bytes
```

Step 3: Activation Bytes (input + output tensors)
```
Input  tensor: 1 × 3 × 224 × 224 = 150,528 elements → 602,112 bytes
Output tensor: 1 × 64 × 112 × 112 = 802,816 elements → 3,211,264 bytes
Activation bytes = (150,528 + 802,816) × 4 = 3,813,376 bytes
```

Step 4: Total Bytes
```
Total bytes = Weight bytes + Activation bytes
            = 37,632 + 3,813,376
            = 3,851,008 bytes
```

Step 5: Arithmetic Intensity
```
AI = FLOPs / Total bytes
   = 236,027,904 / 3,851,008
   ≈ 61.29 FLOP/byte
```




