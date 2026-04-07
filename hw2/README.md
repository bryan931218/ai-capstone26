# AI Capstone HW2

NYCU AI Capstone HW2

Spec: [Google Docs](https://docs.google.com/document/d/1WmlwZPoDuV3adccKl7DBUhWRT2XjSQB81wIH2tBfEIM/edit?pli=1&tab=t.0)

## 1. Preparation
The replica dataset, you can use the same one in `hw0`.

### 2. Environment Setup (Critical)
To avoid **Segmentation Faults** and library conflicts 

- **Python Version**: 3.9 or 3.10 is recommended.
- **NumPy Version**: You **MUST** use `numpy==1.26.4`. (Do NOT use 2.0+).


## Phase 1: Data Collection

Use the Habitat environment to collect RGB-D images and ground truth poses.

### Command

```bash
# Example: Navigating and saving data for the first floor
python load.py
```

##Phase 2: 3D Reconstruction
Switch to the reconstruction environment before running the following commands.
⚠️ Requirements:
open3d
numpy==1.26.4
### Standard Version (Open3D ICP)
Use Open3D's built-in ICP algorithm for reconstruction.
```bash
# Reconstruct Floor 1
python reconstruct.py -f 1

# Reconstruct Floor 2
python reconstruct.py -f 2
```
