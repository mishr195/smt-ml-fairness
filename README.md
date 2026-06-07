# SMT-Backed Machine Learning Fairness Verification Engine

This repository defines the architecture blueprint and component design for an SMT-backed verification engine designed to formally verify individual fairness properties in neural network computation graphs.

## Abstract
Individual fairness requires that similar individuals receive similar outcomes. This engine compiles neural network representations (ONNX or JSON) into mathematical constraint formulas within the Z3 theorem prover. By modeling network behavior and asserting the negation of individual fairness ($|f(x) - f(x')| > \epsilon$), the Z3 solver can mathematically prove model fairness compliance or produce a counter-example exposing discriminatory patterns.

## Installation
Install dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Execution
Run verification using the CLI gateway:
```bash
python src/cli.py --model path/to/model.onnx --preset credit_scoring
```
