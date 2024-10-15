# Baby Cry Classification Model

## Introduction

This repository contains a model for classifying baby cries, specifically designed to monitor and analyze baby sounds for our systems.

## Usage

### Training

To train the model, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/PhoBawBaw/model.git
   cd model
   ```
   
2. Create a new conda environment:
   ```bash
   conda create -n py37 python=3.7
   conda activate py37
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:
   ```bash
   python train.py
   ```

### Serving

To deploy the model for inference, execute the following commands:

1. Clone the repository:
   ```bash
   git clone https://github.com/PhoBawBaw/model.git
   cd model
   ```

2. Build the Docker image:
   ```bash
   sudo docker build -f test.dockerfile -t predict_model .
   ```

3. Run the Docker container:
   ```bash
   sudo docker run -d -p 51213:1213 predict_model
   ```

## Notes

- The model is designed to support inference in environments without GPU.
- Training can also be conducted in CPU-only environments.
- If training on a GPU, ensure that the CUDA settings are compatible with the version of PyTorch being used.
- Access Swagger documentation at `http://0.0.0.0:51213/docs` to explore the API.

## License

TBD

---
