# SpliceDetector

# Overview
SpliceDetector is a deep learning-based tool for identifying splice sitesin DNA sequences. It uses an ensemble of three neural network models with different attention mechanisms to achieve high accuracy in splice site prediction.

## Data availability

The DNA sequence data used for training and testing is sourced from the DRANetSplicer project.You can be downloaded the dataset from https://github.com/XueyanLiu-creator/DRANetSplicer/tree/main/data/dna_sequences

## Installation

### Step 1: Clone the repository

```
git clone https://github.com/HpuBioinformatics/SpliceDetector
cd SpliceDetector
```

### Step 2: Create and activate virtual environment

```
conda create -n SpliceDetector python=3.8
conda activate SpliceDetector
```

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

## Usage

### Training model

```
python train.py \
    --organism_name oryza \
    --splice_site_type donor \
    --train \
    --num_train_epochs 60 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --verbose 1 \
    --report
```

### Prediction 

If you don't want to train from scratch, you can download our pre-trained models in models\trained_models then run predictions directly.

```
python train.py \
    --organism_name oryza \
    --splice_site_type donor \
    --test \
    --batch_size 32 \
    --report
```

Note: Make sure the trained model exists in models/trained_models/ before testing.

## Output Files

```
Training Outputs

models/trained_models/{organism}_{type}_ensemble.pth - Trained ensemble model
results/train_{organism}_{type}_ensemble.txt - Training report

Testing Outputs

results/test_{organism}_{type}_ensemble.txt - Test report
results/pred_probs_{organism}_{type}_ensemble.csv - Prediction probabilities
```

## Complete Example

```
# 1. Train on oryza donor sites
python train.py --organism_name oryza --splice_site_type donor --train --report

# 2. Test the trained model
python train.py --organism_name oryza --splice_site_type donor --test --report

# 3. Check results in the results/ directory
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

