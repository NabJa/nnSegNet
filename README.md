# BraTS21

[Segmenting Brain Tumors in Multi-modal MRI Scans Using a 3D SegNet Architecture](https://link.springer.com/chapter/10.1007/978-3-031-08999-2_32).

## Challange structure
### Task 1: Brain Tumor Segmentation

* Challange Hompage: [here](https://www.med.upenn.edu/cbica/brats2021/)
* Data: [here](https://www.synapse.org/#!Synapse:syn25829067/wiki/610865)

### *Task 2: MGMT Methylation status prediction*
 *Moved to [here](https://gitlab.com/lukassen/glioblastoma/bratc)*

 ## Installation 
 1. Depending on your environment create env (`conda env create --file <file name>`):
    * For cpu: [env_cpu.yml](env_cpu.yml)
    * For gpu: [env_gpu.yml](env_gpu.yml).
 2. Install nnUNet from [here](https://github.com/MIC-DKFZ/nnUNet).
    * Clone repo
    * `pip install -e .` in nnUNet repo
    * `python setup.py install` in nnUNet repo

## Other
### Color codes
- <span style="color:#1f77b4">Edema: #1f77b4</span>
- <span style="color:#ff7f0e">Enhancing tumor: #ff7f0e</span>
- <span style="color:#2ca02c">Necrotic core: #2ca02c</span>
