# mitocheck_td-vae

Using a temporal-difference variational autoencoder [(Gregor et al, 2019)](https://arxiv.org/abs/1806.03107) to predict abnormal cell division phenotypes in the mitocheck dataset ([paper](https://pubmed.ncbi.nlm.nih.gov/20360735/idr), [data](https://idr.openmicroscopy.org/webclient/?show=screen-1101), [metadata](https://github.com/IDR/idr-metadata/blob/master/idr0013-neumann-mitocheck/screenA/)) (implented in [pytorch](https://pytorch.org/)). 

## Data processing  

All mitocheck movies should be downloaded and preprocessed using [this](https://github.com/WayScience/mitocheck_movies) github repo. 
Then, combine all movies into a single pickle file using `convert_videos.py`. 
This script expects that all preprocessesed movies are saved to their orginal output location and have not been moved. 

## Training the model 

`main_train.py` is used to train the model. 
All hyperparameters except `input_size` and `processed_x_size` can be adjusted as desired. 
`input_size` is the dimensions (height * width) of the preprocessed movies and should be changed only if a different compression factor was used during preprocessing. 
Additionally `time_constant_max` and `time_jump_options` should be changed if a different number of frames was specified during preprocessing.
Training the model will generate two outputs. 
The loss for each iteration is saved in `loginfo.txt` and the model is saved every 10 epochs to the `output` folder. 

