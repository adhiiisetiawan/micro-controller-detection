import torch

BATCH_SIZE = 32 
RESIZE_TO = 512 #resize the image for training and transforms
NUM_EPOCHS = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '/home/adhi/micro-controller-detection/data/train/'
VALID_DIR = '/home/adhi/micro-controller-detection/data/test/'

CLASSES = [
    'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]
NUM_CLASSES = 5

OUT_DIR = '/home/adhi/micro-controller-detection/outputs'
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2
