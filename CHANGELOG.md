
# [0.1.1] 12/08/2025
Implemented **padding on a batch** with *collate_fn* PyTorch function.
This way, we train the model with multiple samples by batch. 
This accelerates the training process.

Note: the model is not well trained this way, we now need to define
the right "padding token" id and to ignore it with a mask in the
neural network architecture.
