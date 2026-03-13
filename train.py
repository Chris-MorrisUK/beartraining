from typing import Final

from ultralytics import YOLO
from os import getenv
from pathlib import Path
from loguru import logger

logger.info("Starting training script")
#get the training data path
training_data = Path(getenv("training_data_path","/rds/homes/m/morriscz/custom/custom.yaml"))
if not training_data.exists():
    logger.error("Training data path does not exist: {}", training_data)
    exit(1)
logger.info("Training data path:{}", training_data)
batch_size: Final[int] = int(getenv("BATCH_SIZE", 4))
epochs: Final[int] = int(getenv("EPOCHS", 3))
logger.info("Training parameters: batch size={}, epochs={}", batch_size, epochs)

# Load a model
model = YOLO("yolo26n.pt") 
# Train the model
logger.info("Starting training")
results = model.train(data=training_data, epochs=epochs,  batch=batch_size, imgsz=640,device=[-1, -1])
logger.info("Training complete: {}", results)