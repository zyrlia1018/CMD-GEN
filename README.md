# CMD-GEN

## summary🚀️

**Coarse-grained and Multi-dimensional Data-driven molecular generation (CMD-GEN)**. This framework bridges three-dimensional ligand-protein complex data with two-dimensional drug-like molecule data by utilizing coarse-grained pharmacophore points sampled from diffusion models, thereby enriching the training data for generative models.
Through a hierarchical architecture, it decomposes **the generation of three-dimensional molecules within the pocket into sampling of coarse-grained pharmacophore points**, **generating of chemical structures**, and **alignment of conformations**, avoiding the instability issues associated with inherent in deep generative model-based generation of molecular conformations.

#### TOC graphics

<img src="https://github.com/user-attachments/assets/453a76f4-fdca-4df2-8143-65117cc91c32" width="50%" align="center" />

## How to use :)

### Environment Configuration

To set up the required environment for running the project, follow the steps below for each of the configuration files.

1. Download or clone the `environment_diffphar.yml/environment_gcpg.yml` file from the following link:
   [DiffPhar environment YAML](https://github.com/zyrlia1018/CMD-GEN/blob/main/DiffPhar/env/environment_diffphar.yml)
   [GCPG environment YAML](https://github.com/zyrlia1018/CMD-GEN/blob/main/GCPG/env/environment_gcpg.yml)
2. Create a new Conda environment using the downloaded YAML file:
   
   ```bash
   conda env create -f environment_diffphar.yml
   ```
   
   ```bash
   conda env create -f environment_gcpg.yml
   ```

## pocket-conditioned three-dimensional pharmacophore sampling module

#### 1.Download trained weights from Zenodo : [full-atom & Ca-atom](https://https://zenodo.org/records/13841142)

<div style="text-align: center;">
  <img src="https://github.com/zyrlia1018/CMD-GEN/blob/main/DiffPhar/env/weight.jpg" alt="TOC" width="300" height="100">
</div>

#### 2. Generate Pharmacophore Points

Once you have the trained weights, you can use the `generate_phars.py` script to generate pharmacophore points. This script requires the path to the pre-trained checkpoint, the PDB file for the target protein, and the reference ligand (in the form `chain:index`).
**Command:**

```bash
python generate_phars.py {path_to_pre_trained_checkpoints} --num_nodes_phar 10 --pdbfile {path_to_pdbfile} --ref_ligand {chain:index}
```

Example:

`python generate_phars.py ./checkpoints/best-model-epoch\=epoch\=281.ckpt --num_nodes_phar 10 --pdbfile ./generated/7ONS.pdb --ref_ligand A:1101`

#### 2.1 Perform Dimensionality Reduction and Clustering

To find the important pharmacophore points, we will perform dimensionality reduction using Gaussian Mixture Models (GMM). Follow the steps below:

###### Modify Parameters in `get_phar/GMM_json.py`

Open the `get_phar_GMM_json.py` script and modify the following parameters and Then run this script:

```python
input_file_path = 'phar_to_coords_no_tensor_PARP1.json'  # Modify this to the actual path of your .json file
# Choose the number of clusters (you may need to tune this based on your data)
n_clusters = 7  # You can try values like 5, 6, or 7
```

## Gating Condition Mechanism and Pharmacophore-Based Molecular Generation Module (GCPG)

#### 1.Download trained weights from Zenodo : [checkpoints/](https://https://zenodo.org/records/13841142)

<div style="text-align: center;">
  <img src="https://github.com/zyrlia1018/CMD-GEN/blob/main/GCPG/env/weight2.jpg" alt="TOC" width="300" height="100">
</div>

#### 2. Generate pharmacophore and property-constrained molecules

Currently available properties:：

```
MW, logP, QED, SAS, RotaNumBonds
```

**Pharmacophore types** supported by default:

* AROM: aromatic ring
* POSC: cation
* HACC: hydrogen bond acceptor
* HDON: hydrogen bond donor
* HYBL: hydrophobic group (ring)
* LHYBL: hydrophobic group (non-ring)

The 3d position in `.posp` files will first be used to calculate the Euclidean distances between each point and then the distances will be mapped to the shortest-path-based distances.

###### Modify Parameters in `GCPG/generate.py To cycle through the chemical structures you expect

```
MW_min, MW_max, MW_step = 400, 400, 1
logP_min, logP_max, logP_step = 4, 4, 1
QED_min, QED_max, QED_step = 0.6, 0.6, 1
SAS_min, SAS_max, SAS_step = 4, 4, 1
RotaNumBonds_min, RotaNumBonds_max, RotaNumBonds_step = 4, 4, 1
```

#### 3. Generate

Use the `generate.py` to generate molecules.

usage:

```
python generate.py [-h] [--n_mol N_MOL] [--device DEVICE] [--filter] [--batch_size BATCH_SIZE] [--seed SEED] input_path output_dir model_path tokenizer_path
```

Example:

```
python generate.py data/phar_PARP1.posp gen_result/ result/rs_mapping/fold0_epoch64.pth result/rs_mapping/tokenizer_r_iso.pkl --filter --device cpu
```

The generated results will be saved in {**output_dir**}

## Molecular Binding Conformation Generation Based on Pharmacophore alignment

usage:

```
python generate.py [-smi] [--smiles_path PATH] [--Input_path .psop_file] [--output_fir Path] [--phar_tolerance]
arguments:
  smi           the molecule prepared for association with the pharmacophore aglin
  smiles_path   the input smi file path. ends with `.smi` or `.txt` will be processed
  input_path        the input file path. If it is a directory, then every file ends with `.posp` will be processed
 phar_tolerance      The degree of matching with the pharmacophore
```

We also provide suggested usage see **align.sh** 👍

```
#!/bin/bash

# Path to Python interpreter (adjust as needed)
PYTHON=python3

# Paths and parameters for the Python script
SMILES_PATH="../gen_result/phar_WRN_result.txt"
INPUT_PATH="../data/phar_WRN.posp"
OUTPUT_DIR="./WRN/"
PHAR_TOLERANCE=0

# Check if output directory exists, create if not
mkdir -p "$OUTPUT_DIR"

# Run the Python script
$PYTHON align_test_wrn.py \
    --smiles_path "$SMILES_PATH" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --phar_tolerance $PHAR_TOLERANCE
```

#### Inspiration for this study

Thanks to those works

[https://github.com/CSUBioGroup/PGMG](https://github.com/CSUBioGroup/PGMG)
[https://github.com/arneschneuing/DiffSBDD](https://github.com/arneschneuing/DiffSBDD)
[https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
[https://github.com/biocheming/watvina](https://https://github.com/biocheming/watvina)
[https://github.com/mhlee216/MGCVAE](https://github.com/mhlee216/MGCVAE)
[https://github.com/pengxingang/Pocket2Mol](https://github.com/pengxingang/Pocket2Mol)
[https://github.com/HaotianZhangAI4Science/ResGen](https://github.com/HaotianZhangAI4Science/ResGenE)

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

For commercial use, please contact [zyrlia1018@163.com](zyrlia1018@163.com).


