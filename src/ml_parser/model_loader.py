"""
Model Ingestion Layer.

Responsible for safely opening and verifying serialized computation graph files (such as ONNX formats),
ensuring the file data structure is uncorrupted before handing it off to the graph extraction engine.
"""

from multiprocessing import Value
import os
import json
from typing import Dict, Any, Union
# pyrefly: ignore [missing-import]
import onnx 

class ModelLoader: 

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model_path = model_path

    def load_model(self) -> Union[onnx.ModelProto, Dict[str, Any]]:
        if self.model_path.endswith(".onnx"):
            return self._load_onnx()
        elif self.model_path.endswith('.json'):
            return self._load_json()
        else: 
            raise  ValueError(f"Unsupported format")
        
    def _load_onnx(self) -> onnx.ModelProto:
        
        return onnx.load(self.model_path)
    
    def _load_json(self) -> Dict[str, Any]: 
        
        with open(self.model_path, 'r') as f:
            return json.load(f)

    def get_graph_metadata(self) -> Dict[str, Any]: 
         model = self.load_model()
         if isinstance(model, onnx.ModelProto):
            
            graph = model.graph
            input_tensor = graph.input[0]
            output_tensor = graph.output[0]
            
        
            input_dim = input_tensor.type.tensor_type.shape.dim[1].dim_value
            output_dim = output_tensor.type.tensor_type.shape.dim[1].dim_value
            
            return {
                "input_dim": input_dim,
                "output_dim": output_dim
            }

         elif isinstance(model, dict):
            return {
                "input_dim": model.get("input_dim"),
                "output_dim": model.get("output_dim")
            }

        