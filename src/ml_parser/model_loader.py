"""
Model Ingestion Layer.

Responsible for safely opening and verifying serialized computation graph files (such as ONNX formats),
ensuring the file data structure is uncorrupted before handing it off to the graph extraction engine.
"""
