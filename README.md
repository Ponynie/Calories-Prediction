# Food Classification Project

This repository contains code for a food classification model built using PyTorch Lightning and MobileNetV2 architecture. The model is trained on a dataset of food images from various Thai cuisines.

**Note:** The finished model and a web app for calorie prediction using the model are available in a separate repository: [Calorify](https://github.com/Ponynie/Calorify.git)

You can view the training report of loss, validation metrics, etc. for the model in this project at [https://api.wandb.ai/links/earthcq/iuwopqy8](https://api.wandb.ai/links/earthcq/iuwopqy8).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Customizing the Dataset](#customizing-the-dataset)
- [Updating Checkpoints and Prediction Classes](#updating-checkpoints-and-prediction-classes)
- [Project Structure](#project-structure)
- [Credits](#credits)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ponynie/Popular-Food_Image-Classification.git
   ```

2. Create a new virtual environment (recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. If you encounter any issues with the `torchvision` package, you can install it separately with the appropriate command for your system:

   - On Linux:
     ```bash
     pip install torchvision
     ```

   - On Windows:
     ```bash
     pip install --pre torchvision -f https://download.pytorch.org/whl/nightly/cu117/torch_nightly.html
     ```

   - On macOS:
     ```bash
     pip install torchvision
     ```

## Usage

The project provides several modes for training, testing, analyzing, and predicting with the food classification model.

1. **Training**: To train the model, run the following command:

   ```bash
   python main.py --mode train
   ```

   You can adjust the training hyperparameters by specifying the respective arguments (e.g., `--batch_size`, `--lr`, `--max_epochs`, etc.).

2. **Testing**: To evaluate the model on the test set, run:

   ```bash
   python main.py --mode test
   ```

3. **Analyzing**: To analyze the model's predictions and visualize the attributions, run:

   ```bash
   python main.py --mode analyze
   ```

   Note: In the `analyze_results()` function, you need to replace the image path with the path of the image you want to analyze the attributions for.

   ```python
       image = Image.open('path/to/your/image.jpg').convert('RGB')
       # ... (rest of the function)
   ```

4. **Predicting**: To predict the class of a given image, run:

   ```bash
   python main.py --mode predict --predict_path /path/to/image.jpg
   ```

5. **Exporting**: To export the model for deployment or mobile usage, run:

   ```bash
   python main.py --mode export
   ```

## Customizing the Dataset

To train the model on a new dataset, replace the class folders in the `augmented` directory with your desired class folders. Each class folder should contain images belonging to that class. The class folders should be subfolders of the `augmented` directory.

## Updating Checkpoints and Prediction Classes

If you want to analyze, test, export, or predict using a different trained model checkpoint, you need to update the `checkpoint_path` variable in the respective function (`analyze_results()`, `test_model()`, `export_model()`, or `predict_image()`) to the path of the new checkpoint file. The checkpoint file will be located inside the `MLProject/...` directory after training a new model.

Additionally, if you want to predict classes different from the default ones, you need to update the `food_list` in the `predict_image()` function to match the classes in your new dataset.

## Project Structure

- `main.py`: The main entry point of the project, containing functions for training, testing, analyzing, predicting, and exporting the model.
- `model.py`: Contains the MobileNetV2Lightning model implementation.
- `data_module.py`: Handles data loading and preprocessing for the dataset.
- `image_logger.py`: Implements a custom logger for logging image predictions during training.
- `properties.py`: Stores various configuration properties for the project.

## Credits

This project was developed as part of a machine learning course (2301496) at Chulalongkorn University, Department of Mathematics and Computer Science. Special thanks to the instructors for their guidance and support.

## License

This project is licensed under the [GNU GPLv3 License](LICENSE).
