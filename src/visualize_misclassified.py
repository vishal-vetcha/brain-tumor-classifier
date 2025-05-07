import os
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from model import BrainTumorClassifier  # Import your model class
from torch.utils.data import DataLoader, Dataset
import time  # Import the time module


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label_idx, label in enumerate(self.classes):
            label_path = os.path.join(root_dir, label)
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                self.images.append(file_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def predict_image(image_path, model_path='brain_tumor_classifier.pth'):
    """
    Predicts the class of a brain tumor MRI image.

    Args:
        image_path (str): Path to the MRI image file.
        model_path (str, optional): Path to the trained model's .pth file.
            Defaults to 'brain_tumor_classifier.pth'.

    Returns:
        str: The predicted class of the tumor.
        int: the predicted class index
            Returns "Error" if any issue occurs during prediction.
    """
    try:
        # 1. Load the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Convert to PIL Image

        # 2. Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # 3. Load the model
        model = BrainTumorClassifier(num_classes=4)  # Ensure num_classes matches your training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set the model to evaluation mode

        # 4. Make the prediction
        with torch.no_grad():  # Disable gradient calculation for inference
            img_tensor = img_tensor.to(device)
            output = model(img_tensor)
            _, predicted_class_idx = torch.max(output, 1)
            predicted_class_idx = predicted_class_idx.item()  # Get the integer value

        # 5. Map the class index to the class name
        class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Order must match your training data
        predicted_class_name = class_names[predicted_class_idx]

        return predicted_class_name, predicted_class_idx

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", -1


if __name__ == "__main__":
    # Load the test dataset
    test_dir = 'data/Testing'  # Replace with the path to your testing data
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = BrainTumorDataset(root_dir=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the model
    model = BrainTumorClassifier(num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_path = 'brain_tumor_classifier.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Lists to store misclassified images
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    correct_predictions = 0
    total_images = 0
    start_time = time.time()  # Start timing

    # Iterate through the test set, make predictions, and store misclassified images
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        image_path = test_dataset.images[total_images]
        predicted_class_name, predicted_class_idx = predict_image(image_path)

        if predicted_class_name != "Error":
            if predicted_class_idx != labels.item():
                misclassified_images.append(image_path)
                misclassified_labels.append(labels.item())
                misclassified_predictions.append(predicted_class_idx)
            else:
                correct_predictions += 1
        else:
            print(f"Prediction failed for image: {image_path}")

        total_images += 1
        if total_images % 10 == 0:  # Print progress every 10 images
            elapsed_time = time.time() - start_time
            print(f"Processed {total_images}/{len(test_dataset)} images. Elapsed time: {elapsed_time:.2f} seconds")

    end_time = time.time()  # End timing
    total_time = end_time - start_time

    # Calculate and print the accuracy
    accuracy = correct_predictions / total_images
    print(f"Accuracy on the test set: {accuracy:.4f}")
    print(f"Total time taken: {total_time:.2f} seconds")

    # Visualize misclassified images
    num_misclassified = len(misclassified_images)
    if num_misclassified > 0:
        print(f"\nVisualizing {num_misclassified} misclassified images:")
        plt.figure(figsize=(20, 20))
        for i in range(num_misclassified):
            image_path = misclassified_images[i]
            label = misclassified_labels[i]
            predicted_class_idx = misclassified_predictions[i]
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(17, 17, i + 1)  # changed subplot grid to 17x17
            plt.imshow(img)
            plt.title(f"Actual: {test_dataset.classes[label]}\nPredicted: {test_dataset.classes[predicted_class_idx]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo images were misclassified.")
