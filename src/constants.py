NUM_EPOCHS = 10       # Number of epochs to train
BATCH_SIZE = 18       # Change this depending on GPU memory
NUM_WORKERS = 4       # A value of 0 means the main process loads the data
LEARNING_RATE = 2e-5  # original = 2e-5
LOG_EVERY = 200       # iterations after which to log status during training
VALID_NITER = 600     # iterations after which to evaluate model and possibly save (if dev performance is a new max) -> 30k=600, aug_30k=700, 30K+900K=1100
PRETRAIN_PATH = None  # path to pretrained model, such as BlueBERT or BioBERT
PAD_IDX = 0           # padding index as required by the tokenizer 

# name of 7 observations 
CONDITIONS = ['Cardiomegaly', 'Edema', 'Inspectra Lung Opacity v1',
              'Pleural Effusion', 'Atelectasis', 'Mass', 'Nodule']
# states for each observation
CLASS_MAPPING = {0: "Negative", 1: "Positive"}
