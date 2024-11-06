from zenml.materializers.base_materializer import BaseMaterializer
from sklearn.pipeline import Pipeline
import pickle
import os

class SklearnPipelineMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Pipeline,)
    ASSOCIATED_ARTIFACT_TYPES = ("sklearn_pipeline",)

    def handle_input(self, data_type):
        file_path = os.path.join(self.artifact.uri, "pipeline.pkl")
        with open(file_path, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline

    def handle_return(self, pipeline):
        file_path = os.path.join(self.artifact.uri, "pipeline.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(pipeline, f)
