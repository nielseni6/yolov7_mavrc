# XAI Branch Info

Main script to run is test_plaus_or_faith.py. Plausibility Guided Training implemented (run train_pgt.py). Accuracy of training not yet tested.

# Turret Gunner Survivability and Simulation Environment Project (Main Code)


This project is a collaborative effort between the Machine Learning, AI, and Virtual Reality Center at Rowan University and the Picatinny Arsenal. We, the AI team at Rowan, have been tasked with developing a drone detection algorithm that enhances the survivability of turret gunners. 

Our solution leverages a forked version of YOLOv7, which we have modified to include a special metric for small object detection known as the Normalized Wasserstein Distance. This metric improves the model's ability to detect small objects, a crucial requirement for our simulation environment.

## Setting Up a Virtual Environment

This guide will walk you through the steps to create a new virtual environment, install the required dependencies specified in the `requirements.txt` file.

### Prerequisites

Make sure you have the following installed on your system:

- Python 3.6 or later
- Git
- W&B account (sign up [here](https://wandb.ai/site))

### Creating a Virtual Environment

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/naddeok96/yolov7_mavrc
   ```
2. Navigate to the project directory.
    ```bash
    cd yolov7_mavrc
    ```

3. Create a new virtual environment.
    ```bash
    python3 -m venv venv_name
    ```

4. Activate the virtual environment.
    ```bash
    source venv_name/bin/activate
    ```

## Installing Dependencies

1. Install the dependencies specified in the `requirements.txt` file.
    ```bash
    pip3 install -r requirements.txt
    ```
