import logging
import pickle
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ml_project.config import Config
from ml_project.modelling_process.data_processing import PreprocessingObjects

logger = logging.getLogger('standard')


@dataclass
class ModelArtifacts:

    model: Any
    preprocessing_objects: PreprocessingObjects
    config: Config


def export_model_artifacts(config: Config, model: Any, preprocessing_objects: PreprocessingObjects) -> Optional[ModelArtifacts]:
    """
    Stores model artifacts as a serialised object
    """

    model_artifacts: Optional[ModelArtifacts]
    if config.export_filepath is not None:
        model_artifacts = ModelArtifacts(model=model,
                                       preprocessing_objects=preprocessing_objects,
                                       config=config,
                                       )

        with open(config.export_filepath, "wb+") as file:
            pickle.dump(model_artifacts, file)

        logger.info(f"Model artifacts stored to {config.export_filepath}")
    else:
        model_artifacts = None

    return model_artifacts


def load_model_artifacts(model_objects_filepath: Optional[str]) -> Tuple[Optional[Any], Optional[PreprocessingObjects], Optional[Config]]:
    """
    Loads and returns model artifacts from the provided filepath
    """

    model: Optional[Any]
    preprocessing_objects: Optional[PreprocessingObjects]
    config: Optional[Config]

    if model_objects_filepath is not None:

        with open(model_objects_filepath, "rb") as file:
            model_objects: ModelArtifacts = pickle.load(file)

        model = model_objects.model
        preprocessing_objects = model_objects.preprocessing_objects
        config = model_objects.config
    else:
        logger.warning(f"Export_filepath is None")
        model, preprocessing_objects, config = None, None, None

    return model, preprocessing_objects, config