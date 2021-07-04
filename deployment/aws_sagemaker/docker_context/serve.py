import logging
import os
import sys

import predict

from ml_project.model_export import load_model_artifacts

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("model")


def model_fn(model_dir):
    logger.info("In model_fn()")
    logger.info(os.listdir())
    logger.info(os.listdir(model_dir))
    model_filepath = os.path.join(model_dir, "model_titanic.pkl")
    model, preprocessing_objects, config = load_model_artifacts(model_objects_filepath=model_filepath) #config.export_filepath)

    return (model, preprocessing_objects, config)


def predict_fn(data_array, model_objects):

    # the model is loaded when 'predict' module is imported - could also provide it here via 'model_fn' !
    prediction = predict.predict(data_array, model_objects)

    logger.info("input data: ")
    logger.info(data_array)
    logger.info("Prediction: ")
    logger.info(prediction)

    return prediction