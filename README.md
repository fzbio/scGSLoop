# scGSLoop
Source code for "A lightweight framework for chromatin loop detection at the single-cell level"

## Installation

---

To get started, kindly clone the project repository onto your local machine, navigate to the project directory, and proceed to create a conda environment:

```
git clone https://github.com/fzbio/scGSLoop.git
cd scGSLoop
conda create -n scloop python=3.8
```

Activate the conda environment and install all the requirements:
```
conda activate scloop
pip install -r requirements.txt
```

**Download the five folders (`models`, `data`, `preds`, `refined_scools`, `region_filter`) from [scGSLoop assets](https://portland-my.sharepoint.com/:f:/g/personal/fuzhowang2-c_my_cityu_edu_hk/EurHwCqLAKJGsX7HVfgr6rUBE2ETdY5EmE0myo_oJEu5eg?e=YM5hdc), and copy them to the project directory.**



## Usage:

---

To use scGSLoop to predict loops, the user only needs to modify the fields in `configs.py`. The meanings of the fields in `configs.py` are listed below:

``` text
Possible fields:
    SCOOL_100KB: Path to the .scool file of 100 kb resolution.
    SCOOL_10KB:  Path to the .scool file of 10 kb resolution.
    MODEL_ID:    Identifier of a trained model. 
                 E.g., "mES_k3_GNNFINE" is the model trained on the mES
                 dataset; "hpc_k3_GNNFINE" is the model trained on the
                 hPFC dataset. These identifiers are used for specifying
                 a model in the `models` directory.
    CHROMOSOMES: A Python literal to specify the chromosomes where loops
                 are called from. 
                 Example: ['chr' + str(i) for i in range(1, 23)]
    MODEL_DIR:   The path to the directory where models are stored. 
    OUT_DIR:     The path to the directory to save the predictions.
    THRESHOLD:   A float number as the cutoff to convert the probability 
                 scores to binary predictions. Recommended: 0.5
    MOTIF_FEATURE_PATH:
                 Path to the motif features. We provide motif features for 
                 different assemblies in `data/graph_features`. Now hg19,
                 hg38, mm9, and mm10 are supported. 
    KMER_FEATURE_PATH:
                 Path to the k-mer features. We provide k-mer features for 
                 different assemblies in `data/graph_features`. Now hg19,
                 hg38, mm9, and mm10 are supported. 
    IMPUTE:      A boolean value specifying whether to conduct imputation.
                 We recommend setting this value to False when the median 
                 number of contacts in individual cells exceeds 700,000.
    OUT_IMPUTED_SCOOL_100KB:
                 Path to output the imputed .scool file of 100 kb resolution.
                 This field will be ignored when IMPUTE is set to False.
    OUT_IMPUTED_SCOOL_10KB:
                 Path to output the imputed .scool file of 10 kb resolution.
                 This field will be ignored when IMPUTE is set to False.
    IMPUTATION_DATASET_DIR:
                 Path to the location where the PyTorch dataset for 
                 imputation will be saved. 
                 This field will be ignored when IMPUTE is set to False.
    GENOME_REGION_FILTER:
                 Blacklist regions of the genome assembly. If you don't want
                 to filter the predictions, please set this field to None.
    LOADER_WORKER:
                 Number of workers to load PyTorch dataset. Set to 0 to 
                 work in single-process mode.
```

After configuring the program, run it by:
```
python predict_eval.py
```