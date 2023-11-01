import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import io
import numpy as np 
import tensorflow as tf


STAGE = "Transfer learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_even_odd_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        even_condition = label%2 == 0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels

def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    
    ## get the data 
    (x_train_full,y_train_full),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
    x_valid,x_train =x_train_full[0:5000]/255,x_train_full[5000:]/255
    y_valid,y_train=y_train_full[0:5000],y_train_full[5000:]
    x_test=x_test/255

    y_train_bin,y_test_bin,y_valid_bin = update_even_odd_labels([y_train,y_test,y_valid])

    ## creating seed 
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## log our model summary informatio in logs 
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    ## load the base model 
    base_model_path = os.path.join("artifacts", "models", "base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    
    ## model summary 
    logging.info(f"loaded base model summary: \n{_log_model_summary(base_model)}")

    ## freeze the weights
    for layer in base_model.layers[:-1]:
        print(f"trainable status of before {layer.name}:{layer.trainable}")
        layer.trainable = False
        print(f"trainable status of after {layer.name}:{layer.trainable}")

    base_layer = base_model.layers[:-1]

    ## define the model and compile it 
    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(
        tf.keras.layers.Dense(2,activation="softmax",name="output_layer")
    )
    ## model summary 
    logging.info(f"loaded base model summary: \n{_log_model_summary(new_model)}")

    LOSS_FUNCTION ="sparse_categorical_crossentropy"
    OPTIMIZER= tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS =["accuracy"]

    new_model.compile(loss=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS)

    EPOCHS=30
    VALIDATION = (x_valid,y_valid_bin) # y_valid_bin for our use cases 

    history = new_model.fit(x_train,y_train_bin,validation_data=VALIDATION,epochs=EPOCHS,verbose=2)

    ## save the model 
    model_dir_path = os.path.join("artifacts","models")

    model_file_path = os.path.join(model_dir_path, "even_odd_model.h5")
    new_model.save(model_file_path)

    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {new_model.evaluate(x_test,y_test_bin)}")




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e