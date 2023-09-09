import subprocess
import os
import shutil
import logging
import cv2
import torch
import yaml

# Configure logging to display INFO level logs
logging.basicConfig(level=logging.INFO)


def download_yolov5_repository():
    """
    Downloads the YOLOv5 repository from GitHub if it does not already exist.

    Returns:
        None
    """
    yolov5_path = os.path.join(os.getcwd(), "yolov5")

    # Check if the Yolov5 directory already exists
    if not os.path.exists(yolov5_path):
        print("YOLOv5 repository not found. Downloading...")

        # Use git to clone the YOLOv5 repository from GitHub
        try:
            subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", yolov5_path])
            print(f"YOLOv5 repository downloaded successfully to '{yolov5_path}'.")
        except Exception as e:
            print(f"Error downloading YOLOv5 repository: {e}")
    else:
        print(f"YOLOv5 repository is already present in '{yolov5_path}'.")


class YoloTrainer:
    def __init__(self):
        self.temp_dir = "temp"  # Temporary directory
        self.output_model = "best.pt"
        self.result_folder = os.path.join(os.getcwd(), "yolov5", "runs", "train", "yolov5s_results")
        self.output_model_path = os.path.join(self.result_folder, "weights", self.output_model)

        self.start()

    def start(self):
        """
        Start the process by asking the user for the destination path of the dataset
         and then run the process with the provided destination path.
        Returns:
            None
        """
        destination_path = self.ask_path("the dataset")
        self.run(destination_path)

    def menu(self):
        """
        Displays a menu with options and executes the corresponding action based on user input.

        Returns:
            None
        """
        while True:
            print(self.wrap_in_step("\nOptions:\n"
                                    "1 - Copy the result folder (model and all its generated data)\n"
                                    "2 - Copy only the model\n"
                                    "3 - Test the model\n"
                                    "4 - Generate another model\n"
                                    "5 - Exit"))

            choice = input("Enter your choice (1/2/3/4): ")

            if choice == "1":
                self.copy_result_folder()
            elif choice == "2":
                self.copy_only_model()
            elif choice == "3":
                self.test_model()
            elif choice == "4":
                self.start()
            elif choice == "5":
                print(self.wrap_in_step("Exiting the program."))
                return
            else:
                logging.log(logging.INFO, "Invalid choice. Please select a valid option.")

    def prepare_dataset(self, dataset_path):
        """
        Prepares the dataset by copying the contents of the provided dataset_path to the temp/dataset folder.

        Args:
            dataset_path (str): The path to the dataset that needs to be copied.

        Returns:
            str: The path where the dataset was copied.
                Returns `None` in case of an error.
        """
        # Ensure the temp/dataset folder does not exist before creating it
        temp_dataset_path = os.path.join(self.temp_dir, "dataset")
        if os.path.exists(temp_dataset_path):
            shutil.rmtree(temp_dataset_path)

        # Copy the contents of the provided dataset_path to the temp/dataset folder
        try:
            shutil.copytree(dataset_path, temp_dataset_path)

            print(self.wrap_in_step(f"Dataset copied to {temp_dataset_path}"))
            return temp_dataset_path  # Return the path where the dataset was copied
        except Exception as e:
            logging.log(logging.ERROR, f"Error copying dataset: {e}")
            return None  # Return None in case of an error

    def copy_result_folder(self):
        """
        Copies the result folder to a specified destination path.

        Returns:
            None
        """
        destination_path = self.ask_path("copy the result folder")
        destination_path = os.path.join(destination_path, "trained_model_data")

        try:
            shutil.copytree(self.result_folder, destination_path)
            print(self.wrap_in_step(f"Result folder copied to {destination_path}"))
        except Exception as e:
            logging.log(logging.ERROR, f"Error copying result folder: {e}")

    def copy_only_model(self):
        """
        Copy the trained model to a specified destination folder.

        Returns:
            None
        """
        destination_folder = self.ask_path("copy the model")

        # Create the destination path with the new name
        destination_path = os.path.join(destination_folder, "trained_model.pt")

        try:
            shutil.copy(self.output_model_path, destination_path)
            print(f"Model copied to {destination_path}")
        except Exception as e:
            logging.log(logging.ERROR, f"Error copying model to destination: {e}")

    def run(self, dataset_path):
        """
        Run the program using a given dataset.

        Args:
            dataset_path (str): The path to the dataset.

        Returns:
            None
        """
        name = "yolov5s_results"
        # Step 1: Prepare the dataset in a temporary directory
        copied_path = self.prepare_dataset(dataset_path)

        # Step 2: Modify the data.yaml file (if needed)
        if copied_path is not None:
            modified_data_yaml = os.path.join(copied_path, "data.yaml")
            self.modify_data_yaml(modified_data_yaml)
            print(self.wrap_in_step(f"Modified data.yaml path: {modified_data_yaml}"))

            print(self.wrap_in_step("Now, we need a few params to be setted."))

            cfg_path = self.ask_path("the .yaml file of the model (ex: models/yolov5s.yaml or full path)")
            weights_path = self.ask_path("the .pt file for the weight of the model (ex: yolov5s.pt or full path)")

            # Input values for img_size, batch_size, and epochs
            img_size = int(input("Enter the image size (e.g., 416): "))
            batch_size = int(input("Enter the batch size (e.g., 16): "))
            epochs = int(input("Enter the number of epochs (e.g., 5): "))

            # Step 3: Train the model
            self.train(img_size, batch_size, epochs, modified_data_yaml, cfg_path, weights_path, name)

            print(self.wrap_in_step("Train done. Now, the model is ready to be used."))

            self.menu()

            # Final step: Delete the temporary folders (dataset and trained_model)
            shutil.rmtree(os.path.join(self.temp_dir))
        else:
            logging.log(logging.INFO, "Something went wrong copying the dataset. Path is correct ?")
            self.start()

    @staticmethod
    def wrap_in_step(text):
        """
        Generates a text layout by wrapping the provided text in borders.

        Parameters:
            text (str): The text to be wrapped in the step menu.

        Returns:
            str: The step menu with the text wrapped in borders.
        """
        lines = text.splitlines()
        max_length = max(len(line) for line in lines)

        # Generate the top and bottom borders
        top_border = f"+{'-' * (max_length + 2)}+"
        bottom_border = top_border

        # Generate the content with '|' borders
        content = "\n".join(f"| {line.ljust(max_length)} |" for line in lines)

        # Combine the borders and content
        step_menu = f"{top_border}\n{content}\n{bottom_border}"

        return step_menu

    @staticmethod
    def ask_path(what=""):
        """
        Asks the user for a destination path.

        Parameters:
            what (str): The specific purpose of the destination path.

        Returns:
            str: The full destination path entered by the user.
        """
        for_what = ""
        if what != "":
            for_what = f" for {what}"
        destination_path = input(f"Enter the full destination path{for_what}: ").replace('"', "")
        return destination_path

    @staticmethod
    def modify_data_yaml(data_yaml_path):
        """
        Modifies the data.yaml file by updating the necessary paths in the data dictionary.

        Parameters:
            data_yaml_path (str): The path to the data.yaml file.

        Returns:
            None
        """
        # Load the existing data.yaml file
        with open(data_yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        # Modify the necessary paths in the data dictionary
        data['train'] = "../temp/dataset/train/images"
        data['val'] = "../temp/dataset/valid/images"

        # Save the updated data dictionary back to the file
        with open(data_yaml_path, 'w') as file:
            yaml.dump(data, file)

    @staticmethod
    def get_advanced_options():
        """
        Displays a menu of available training options and collects user selections.

        Returns:
            dict: A dictionary containing user-selected options and their values.
        """
        options = {}

        # Ask the user if they want to apply advanced parameters
        apply_advanced = input("Apply advanced training parameters (if in doubt, say 'n') (y/n)? ").strip().lower()
        if apply_advanced == "y":
            print("\nSelect advanced training options (press Enter to skip):\n")

            # Define the available advanced options and their descriptions
            advanced_options = {
                "--lr": "Learning rate",
                "--multi-scale": "Enable multi-scale training",
                "--sync-bn": "Use synchronized batch normalization",
                "--cache-images": "Cache images for faster training",
                # Add more options here as needed
            }

            for option, description in advanced_options.items():
                user_input = input(f"Include {description} ({option}) (y/n)? ").strip().lower()
                if user_input == "y":
                    if option in ("--sync-bn", "--cache-images"):
                        value = True  # For boolean options, set to True if selected
                    else:
                        value = input(f"Enter the value for {option} (e.g., 0.001): ").strip()
                    options[option] = value

        return options

    def train(self, img_size, batch_size, epochs, data_yaml, cfg_path, weights_path, name):
        """
        Trains the model using the specified parameters with user-selected options.
        Uses subprocess to do the training.

        Args:
            img_size (int): The size of the input images.
            batch_size (int): The number of samples per gradient update.
            epochs (int): The number of times the entire dataset is traversed during training.
            data_yaml (str): The path to the YAML file containing the dataset information.
            cfg_path (str): The path to the configuration file.
            weights_path (str): The path to the pre-trained weights file.
            name (str): The name of the training session.

        Returns:
            None
        """

        yolov5_path = os.path.join(os.getcwd(), "yolov5")
        runs_path = os.path.join(os.getcwd(), "yolov5", "runs")
        if os.path.exists(runs_path):
            shutil.rmtree(runs_path)

        final_data_yaml = os.getcwd() + "/" + data_yaml
        final_cfg_path = yolov5_path + "/" + cfg_path

        # Get user-selected options
        user_options = self.get_advanced_options()

        # Construct the final training command with user-selected options
        final_command = (f"python {yolov5_path}/train.py --img {img_size} --batch {batch_size} "
                         f"--epochs {epochs} --data {final_data_yaml} --cfg {final_cfg_path} "
                         f"--weights {weights_path} --name {name} --cache")

        for option, value in user_options.items():
            if value is not None:
                final_command += f" {option} {value}"

        print(f"\nStarting training with command: {final_command}")

        try:
            # Spawn a new process for the final training command
            final_process = subprocess.Popen(
                final_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # Line buffered
                universal_newlines=True,  # Text mode
            )

            try:
                # Process and print the output line by line in real-time
                for line in final_process.stdout:
                    print(line, end='')
            except Exception as e:
                print(f"Error during training: {e}")

            # Wait for the final process to complete
            final_process.wait()
        except subprocess.CalledProcessError as e:
            logging.log(logging.ERROR, f"Error during training: {e}")

    @staticmethod
    def test_model():
        """
        Runs a test on a given image path.

        Returns:
            None
        """
        image_path = input("Enter the full path of the image to test: ")

        model = torch.hub.load('yolov5',
                               'custom',
                               path='yolov5/runs/train/yolov5s_results/weights/best.pt',
                               source='local')
        # Set the model to evaluation mode
        model.eval()

        # Load an image
        image = cv2.imread(image_path)

        # Perform inference
        results = model(image)

        # Get the detected objects
        predictions = results.pred[0]  # Use [0] for the first image in the batch (if processing multiple images)

        # Access bounding boxes, confidence scores, and class labels
        boxes = predictions[:, :4]  # Bounding boxes (x1, y1, x2, y2)
        scores = predictions[:, 4]  # Confidence scores
        labels = predictions[:, 5]  # Class labels

        # Loop through the detected objects
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box  # Coordinates of the bounding box
            class_name = model.names[int(label)]  # Convert label index to class name

            # Draw bounding box and label on the image
            color = (0, 255, 0)  # Green color for the bounding box
            thickness = 2  # Thickness of the bounding box lines
            font_scale = 0.5  # Font scale for the label
            font_thickness = 1  # Thickness of the font
            text = f"{class_name}: {score:.2f}"  # Label text

            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), color, cv2.FILLED)
            cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), font_thickness)

        # Display the image with detections
        cv2.imshow('Detections', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    download_yolov5_repository()
    YoloTrainer()
