# MEBA-for-Crowd-Counting

The above code are divived into two files, SHHA and SHHB for corresponding ShanghaiTechA and ShanghaiTechB datasets.



## Download Dataset [Link](https://www.kaggle.com/datasets/tthien/shanghaitech)



## Train and Test

1. Pre-Process Data (resize image and split train/validation) (This step has been done in the sha-Train-Val-Test/shb-Train-Val-Test file, but you can try again by youself.)
```
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>
```

2. Train Model
```
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>
```

3. Test Model
```
python test.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```



## Pretrained Models

ShanghaiTechA: [Link](https://drive.google.com/drive/folders/1xolV-c8l1IUVDr6Wq8mel7K8bD6SJqbr?usp=sharing)

ShanghaiTechB: [Link](https://drive.google.com/drive/folders/1aT53Nco3d6Z7ejEbPSt2TdwAv-5dmJfo?usp=sharing)
