from .models import FCN, load_model
from .utils import SuperTuxDataset, ConfusionMatrix

# TODO list
'''
1. Move Torch.Transforms out of the __init__, do we want to make it configurable through the command line argument?
2. Simplify model while maintaining highest accuracy
3. Data Augmentation can be randomized on training set for highest generalization (no/less augmentation on validation)
4. Try out Dropout layer (where applicable), Normalization techniques (namely batch), residual connection, etc to get accuracy up
5. Simplify Tensorboard utilization 
6. FCN model work 

FCN work:
1. apply Dense transforms to image
2. See if CUDA is enabled on PC
3. Set up environment on my Mac for training model 
4. Read architecture of u-net and see what I can derive from it 
5. write training code for FCN 
6. Set up model 
'''