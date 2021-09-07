# Kaggle_30DaysOfML

https://www.kaggle.com/c/30-days-of-ml/leaderboard

Instruction for running in google colab
```buildoutcfg
!git clone https://github.com/user-vi/Kaggle_30DaysOfML.git
!pip install -r ./Kaggle_30DaysOfML/requirements.txt
```
restart runtime

```buildoutcfg
!pip install pydrive2
%cd ./Kaggle_30DaysOfML/
!ls
!dvc pull
!dvc repro
```