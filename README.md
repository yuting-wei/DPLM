# DPLM: A Diachronic Language Model for Long-Time Span Classical Chinese

This repository provides a reference implementation of the following paper:

> **A Diachronic Language Model for Long-Time Span Classical Chinese**  
> Yuting Wei, Meiling Li, Yangfu Zhu, Yuanxing Xu, Yuqing Li, Bin Wu  
> *Information Processing and Management*

## Prerequisites

To set up the environment, follow these steps:

1. Create an Anaconda environment with Python 3.8:

    ```bash
    conda create -n temporal python=3.8
    conda activate temporal
    ```

2. Install the required PyTorch version and dependencies:

    ```bash
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    ```

3. Install the remaining Python packages:

    ```bash
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```
    
4. Download our training dataset:

    Please download the dataset from Google Drive [here](https://drive.google.com/file/d/1EQT6-kQ8p8CfA7JbztmytSPzH8bF_Xm8/view?usp=sharing) and place it in the `train_datasets` folder.

## Training

To train the model, simply run the following command:

```bash
bash run.sh
```

## Evaluation

To evaluate the model, use the following command, replacing `path_to_trained_dplm` with the path to your trained model:

```bash
python evaluate.py --model_name path_to_trained_dplm
```

## Citation

Please cite our paper if you use our code, dataset, or compare with our model:
```
@article{WEI2025103925,
title = {A diachronic language model for long-time span classical Chinese},
journal = {Information Processing & Management},
volume = {62},
number = {1},
pages = {103925},
year = {2025},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2024.103925},
url = {https://www.sciencedirect.com/science/article/pii/S030645732400284X},
author = {Yuting Wei and Meiling Li and Yangfu Zhu and Yuanxing Xu and Yuqing Li and Bin Wu}
}
```

## Acknowledgements

The code structure is inspired by [TempoBERT](https://github.com/guyrosin/tempobert/tree/main). The eval datasets are adapted from the [Ancient Chinese Corpus with Word Sense Annotation](https://github.com/iris2hu/ancient_chinese_sense_annotation). We sincerely thank the authors of these projects for their contributions to the community.
