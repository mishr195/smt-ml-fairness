"""
Computation Graph Node Extractor.

Traverses the ingested model structure to dynamically parse tensor shape boundaries, isolating raw 
floating-point weight matrices and corresponding bias vectors for every hidden layer without hardcoding.
"""
