# GRC-ProbNet: Uncertainty-aware feature extraction for Cardiovascular Disease Classification

This repository builds upon the original **GRC-Net** framework:

> A. Mittal, R. Mehta, O. Todd, P. Seeboeck, G. Langs, B. Glocker  
> *Cardiovascular disease classification using radiomics and geometric features from cardiac CT images*

We gratefully acknowledge the original authors for their codebase, ideas, and foundational work, which this project extends.

---

## Project Overview

This project is the implementation of a BEng thesis on uncertainty propagation within cardiac pipelines. The main goal is to investigate how **epistemic and ensemble-based uncertainty estimation** can improve robustness and interpretability in medical imaging pipelines.

### Key Contributions of This Work

This project extends the original GRC-Net framework with:

- **Uncertainty-aware feature extraction pipelines**
  - Multi-seed and ensemble-based uncertainty estimation
  - Variance- and entropy-based feature construction

- **Ensemble learning approaches**
  - Prediction-space ensembles
  - Feature-space aggregation strategies
  - Weighted and inverse-variance fusion methods

- **Coronary artery pipeline adaptation**
  - Extension of the original cardiac structure pipeline to coronary segmentation
  - Dedicated uncertainty propagation experiments for coronary datasets

- **Comprehensive experimental analysis**
  - Ablation studies across multiple uncertainty formulations
  - Multi-fold evaluation on ASOCA dataset

---

## Code Structure

Each experimental pipeline generally follows a consistent structure:

> **Fine-tuning → Atlas-ISTN registration → Uncertainty computation (optional) → Classification**

For example:

`anatomix-fine-tuning.py → atlas-istn-anatomix.py → compute_uncertainty.py (+ aggregation scripts) → Geo-Radio-Classification.py`

The classification step requires:
- A trained Anatomix segmentation model (from step 1)
- A trained registration model and constructed atlas (from step 2)  
The required atlas labelmaps will be automatically generated during the registration stage.

File names vary depending on the specific experimental setup, but the overall pipeline structure remains consistent across both cardiac and coronary experiments.

---

The repository is organised into the following main components:

### `cardiac_structure_files/`
Contains experiments for the **original cardiac structure classification pipeline**, extended with uncertainty-aware modelling.

- Multi-seed classification experiments
- Uncertainty propagation methods
- Ensemble prediction pipelines
- Atlas-ISTN based feature extraction experiments

---

### `coronary_files/`
Contains the adapted pipeline for **coronary artery segmentation and classification**.

- Coronary-specific segmentation and classification models
- Uncertainty-aware coronary experiments
- Ensemble and variance-based aggregation methods
- Evaluation scripts for coronary disease classification


---

### `utils/`
Contains helper functions for visualation of results, metrics computation and other pipeline utilities

---

### `img/`
Core data processing module:

- Dataset loading (`datasets.py`)
- Image preprocessing (`processing.py`)
- Transformations and augmentation pipelines (`transforms.py`)

---

### `nets/`
Neural network architectures used in the pipeline:

- Convolutional models
- Spatial transformer networks (STN)
- Gaussian convolution modules

---

## Data Setup

This project assumes the following datasets are placed in the `data/` directory:

- ASOCA dataset
- MM-WHS dataset

### ASOCA

- Images → `data/ASOCA/images`
- Labels → `data/ASOCA/labels`

### MM-WHS

- Images → `data/MM-WHS/images`
- Labels → `data/MM-WHS/labels`

---

## Data Splits

The ASOCA dataset uses a predefined 5-fold cross-validation setup provided in:

- `data/config/asoca`

Each fold contains:
- 32 training subjects
- 8 held-out test subjects

This split structure is used across all coronary experiments to ensure reproducibility.

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3.9 conda environment:

   ```shell
   conda create -n grcnet python=3.9
   conda activate grcnet
   ```
Install PyTorch using conda (for CUDA toolkit 12.4)
   ```shell
    # torch dependencies
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

    # other Python packages
    pip install numpy==1.23 torchio tensorboard tensorboardX

    # TotalSegmentator and utils
    pip install TotalSegmentator
    pip install --upgrade acvl_utils==0.2

    # GPU transforms (MONAI with CuCIM)
    pip install monai[cucim] cupy-cuda12x==12.3.0

    # Patches
    pip install mkl==2024.0 dicom2nifti==1.2.21 pydicom==1.4.1
  ```
   
To run both the Anatomix fine-tuning and Geo-Radio Classification tutorial:
```shell
  pip install ipykernel ipywidgets pyradiomics optuna
```

### For future Imperial students/staff

To submit SLURM scripts and run this pipeline within the Imperial BioMedIA cluster:

```shell
ssh <username>@biomedia-slurm
cd /vol/biomedic2/<wherever your grc-probnet is cloned to>
sbatch <script>
squeue -u <username>  # check status
```

To activate an environment and run a Python file:

```shell
source /vol/biomedic2/<path to where your miniconda3 is>/miniconda3/bin/activate
conda activate grcnet
python <filename>
```

To check the status of other machines using the SLURM cluster:

```shell
/vol/biomedic3/bin/lazyslurm
```

To view your SLURM log files:

```shell
ls -lt slurm*
```

## Reproducibility Notes

- Experiments may take overnight or longer depending on GPU availability
- Recommended hardware: 24GB+ GPU memory
- Small variations may occur due to randomness in training and ensemble sampling
- Ensure consistent dataset splits for reproducibility

## License
This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgment
This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement 101080302 (AI-POD).
