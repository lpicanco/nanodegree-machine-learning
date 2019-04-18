# Machine Learning Engineer Nanodegree
## Facial Expression Recognition

This is my Machine Learning Engineer Nanodegree Capstone project.

[Jupyter Notebook with the CNN training and final results](https://github.com/lpicanco/nanodegree-machine-learning/blob/master/capstone/facial_expression_recognition.ipynb)

### How to train the network:

```bash
cd src
python train.py 
```

### How to predict an emotion using a trained model:
```bash
python src/predict.py  model/cnnmodel-095-0.7827-0.7365.hdf5 sample_images/angry01.jpg
```

### Sample results:
![sample_results](https://raw.githubusercontent.com/lpicanco/nanodegree-machine-learning/master/capstone/imgs/faces_prediction01.png)
