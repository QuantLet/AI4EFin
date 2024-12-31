# Energy Price Modelling: A Comparative Evaluation of four Generations of Forecasting Methods 

* This is the official github repository for the paper benchmarking four generations of time series models for long-term forecasting of electricity price data. [[paper](https://arxiv.org/abs/2411.03372)]

## Key Results
The models PatchTST and TimesFM from the 3. generation (Transformers) and the 4. generation (Pre-trained Models) outperform on average all models included in our benchmarking study on the long-term forecasting of electricity price data.

## Reproducibility
1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from ```./data/data_import/```.

3. Training the models. You can use the code in the jupyter notebook ```./src/Run_TS_Models.ipynb``` to train each time series model included in our study and to test its performance on unseen part of the multivariate time series data. The notebook imports all the necessary modules from ```./time_series_models/```. 

4. Evaluation of the performance. You can use the code in the jupyter notebook ```./src/Evaluate_and_Visualize_Forecasts.ipynb``` to evaluate the generated forecasts and to reproduce the visualizations in our study. 


## Acknowledgement

We appreciate the following github repositories very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/yuqinie98/PatchTST

https://github.com/nzl5116190/Basisformer

https://github.com/DAMO-DI-ML/KDD2022-Quatformer

https://github.com/google-research/google-research/tree/master/tsmixer



## Contact

If you have any questions or concerns, please contact us: andrei1victor23@stud.ase.ro, velegeor@hu-berlin.de, toma.filip.mihai@gmail.com or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@article{Andreietal-2024-FourGenerations,
  title     = {Energy Price Modelling: A Comparative Evaluation of four Generations of Forecasting Methods},
  author    = Andrei, Alexandru-Victor  and
              Velev, Georg and 
              Toma, Filip-Mihai and 
              Pele, Daniel Traian  and 
              Lessmann, Stefan},
  journal = {arxiv preprint arXiv:2411.03372},
  year      = {2024}
}
```
