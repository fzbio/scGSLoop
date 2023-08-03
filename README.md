# scGSLoop
Source code for "A lightweight framework for chromatin loop detection at the single-cell level"

## Installation


1. To get started, please clone the project repository onto your local machine, navigate to the project directory, and proceed to create a conda environment:

    ```
    git clone https://github.com/fzbio/scGSLoop.git
    cd scGSLoop
    conda create -n scloop python=3.8
    ```

2. Activate the conda environment:

    ```
    conda activate scloop
    ```

3. **Download the five folders (`models`, `data`, `preds`, `refined_scools`, `region_filter`) from [scGSLoop assets](https://portland-my.sharepoint.com/:f:/g/personal/fuzhowang2-c_my_cityu_edu_hk/EurHwCqLAKJGsX7HVfgr6rUBE2ETdY5EmE0myo_oJEu5eg?e=YM5hdc), and copy them to the project directory.**

4. Install PyTorch >= 1.8.0 according to its official [documentation](https://pytorch.org/get-started/previous-versions/). We recommend using [PyTorch 1.8.*](https://pytorch.org/get-started/previous-versions/#linux-and-windows-23) for best compatibility.

5. Install PyTorch-Geometric:
   ```
   conda install pyg -c pyg -c conda-forge
   ```

6. Install other dependencies using the following command:

    ```
    pip install -r requirements.txt
    ```

---


## Usage:

### Predict at the single-cell level

ScGSLoop accepts `.scool` files as input. If this format sounds unfamiliar to you, kindly check out [Cooler](https://github.com/open2c/cooler)'s [documentation](https://cooler.readthedocs.io/en/latest/api.html#cooler.create_scool) for detailed descriptions. 

The user needs to prepare their data of two resolutions: 10 kb and 100 kb. If you only have the resolution of 10 kb, you can simply coarsen it using [cooler coarsen](https://cooler.readthedocs.io/en/latest/cli.html#cooler-coarsen).

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

The loop calls of each cell at the single-cell level will be available in the directory you designated as `OUT_DIR`.


### Consensus loop list

After the single-cell loops are detected, you can use them to generate the consensus loop list.

**Note**: In this step, the predictions in `pred_dir` must be of the same cell type.

usage: 
```
python consensus.py [-h] [-p PERCENTILE | -n NUM_LOOP] raw_scool_path pred_dir out_path

Arguments:
   raw_scool_path:   Path to the raw 10kb .scool file
   pred_dir:         Path to the single-cell predictions
   out_path:         Path to output the consensus list
  
Options:
The following two options are mutually exclusive. Choose one of them to set the threshold 
for generating loops.
   -p, --percentile: Percentile among all loop scores. Loops with score ranking higher than 
                     the percentile will be added to the consensus list
   -n, --num-loop:   The total number of loops.
   
Percentiles used in our study: 
   hpc_k3_GNNFINE: 97.35
   mES_k3_GNNFINE: 98.5 
You can adjust the percentile or num loop if there are too many or too few loops in the 
final list.
```

### Hub discovery

Modify the variables in `hub_discover.py`:

```text
chroms:              A list containing the names of desired chromosomes
gene_coords_path:    Path to a csv file containing these columns: 
                     chr,start,end,strand,gene_id,gene_symbol
pred_dir:            Path to the directory of single-cell loop preds
assembly_size:       Path to the assembly size file (e.g. hg19.sizes)
consensus_path:      Path to the consensus loop list
output_path:         Path to the output file 
```
usage:
```
python hub_discover.py
```
 
