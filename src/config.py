import torch

BATCH_SIZE = 64 
RESIZE_TO = 512 #resize the image for training and transforms
NUM_EPOCHS = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '../data/train/'
VALID_DIR = '../data/test/'

CLASSES = [
    'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]
NUM_CLASSES = 5

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2
