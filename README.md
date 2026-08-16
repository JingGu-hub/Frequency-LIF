# TOE: Tackling Distribution Shifts in Time Series via Three-Perspective Online Ensemble

This is the official code for our paper "TOE: Tackling Distribution Shifts in Time Series via Three-Perspective Online Ensemble".

## Abstract

Persistent distribution shift poses a challenging problem in multivariate long-term time series forecasting (MTSF) tasks. Although existing methods have made considerable progress, they still encounter two key limitations. First, methods based on offline learning typically utilize a static set of parameters to model the entire time series. This static approach has limited capacity to adapt to the complex distribution shifts inherent in time-series data. Second, the learning processes of existing online methods are often confined to a single perspective, whereas distribution shifts may manifest across multiple dimensions simultaneously, such as time, frequency, and space. A single-perspective focus can make the model less sensitive to these multifaceted shifts and may lead to conflicts between dimensional adaptations. To address these challenges, this paper proposes a multi-perspective online ensemble framework, the Trio Online Ensemble framework (TOE). TOE consists of three branches—time, frequency, and space—and its core mechanism is summarized as follows: (1) During the training phase, TOE independently maintains a memory pool for each branch. As new data arrive, TOE assesses the current model's suitability for these data via memory retrieval, which in turn informs the decision of whether to trigger branch retraining, thereby enabling dynamic adaptation to distribution shifts. (2) During the testing phase, TOE leverages the test sample from the previous time step to perform memory retrieval. This process indirectly matches the current test sample with suitable model parameters within each of the three branches. We implement a foundational version of TOE (TOE-Base) built upon a fully linear architecture and conduct comprehensive experimental evaluations on eight real-world and eight synthetic datasets. The experimental results show that TOE-Base achieves competitive or better performance than several advanced MTSF methods. Furthermore, applying the TOE framework to other existing methods also yields consistent performance improvements.

## Overall Architecture

Overview of the TOE framework. After offline warm-up, the time, frequency, and space branches execute parallel memory retrieval, conditional retraining, and dynamic ensemble for online adaptation. Architecture of TOE-Base and its three branches. I. Time branch: Reshapes sequences into 2D structures, combining gating mechanisms and 1D convolutions to extract parallel long- and short-term features; II. Frequency branch: Applies frequency-domain linear transformations, using the energy spectrum to extract and broadcast dominant frequencies; III. Space branch: Employs a star aggregation-redistribution mechanism to generate and fuse a global core vector with original features to model spatial dependencies.

<p align="center">
<img src="./figs/online_main.png" alt="online_main" align=center />
</p>
<p align="center">
<img src="./figs/branch_main.png" alt="branch_main" align=center />
</p>

## Datasets

### Eight real-world datasets

* [ETT](https://github.com/zhouhaoyi/ETDataset/tree/main)
* [Traffic, Electricity, Weather](https://github.com/thuml/Autoformer)
* [Exchange](https://www.kaggle.com/datasets/giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets/#site-content)

## Usage
Install Pytorch and the necessary dependencies.

```
pip install -r requirements.txt
```

To train a TOE model on a dataset, run

```bash
python main.py --data ETTm2 --root_path dataset_address --data_path dataset.csv --seq_len 96 --pred_len 96
```

## Citation

If you use this code for your research, please cite our paper:
```
@article{zhou2026toe,
  title={TOE: Tackling Distribution Shifts in Time Series via Three-Perspective Online Ensemble},
  author={Zhou, Yujue and Liu, Jiarong and Peng, Hong and Yang, Shuanghua and Du, Bowen},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026},
  publisher={IEEE}
}
```
