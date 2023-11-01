import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np 
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import io


STAGE = "Creating binary base model from strach " ## <<< change stage name 

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

    ## set the seed 
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## define the model 
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28,28],name="input_layer"),
        tf.keras.layers.Dense(300,name="first_hidden_layer"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(100,name="second_hidden_layer"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(2,activation="softmax", name="output_layer"),
    ]

    ## define the model and compile it 
    model = tf.keras.models.Sequential(LAYERS)
    LOSS_FUNCTION ="sparse_categorical_crossentropy"
    OPTIMIZER= tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS =["accuracy"]

    model.compile(loss=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS)

    
    
    ## log our model summary informatio in logs 
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    ## model summary 
    logging.info(f"binary_scratch_model model summary: \n{_log_model_summary(model)}")


    ## Train the model 

    EPOCHS=30
    VALIDATION = (x_valid,y_valid_bin)
    history = model.fit(x_train,y_train_bin,validation_data=VALIDATION,epochs=EPOCHS,verbose=2)

    ## save the model 
    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "binary_scratch_model.h5")
    model.save(model_file_path)

    logging.info(f"binary_scratch_model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {model.evaluate(x_test,y_test_bin)}")



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