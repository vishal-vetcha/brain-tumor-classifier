import gradio as gr
from PIL import Image
import torch
from torchvision import transforms
from model import BrainTumorClassifier  # Import your model class
import cv2

def predict_image(image):
    """
    Predicts the class of a brain tumor MRI image.

    Args:
        image (PIL.Image.Image): The MRI image to predict.

    Returns:
        str: The predicted class of the tumor.
    """
    try:
        # 1. Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = preprocess(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # 2. Load the model
        model = BrainTumorClassifier(num_classes=4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model_path = 'brain_tumor_classifier.pth'  # Path to your trained model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 3. Make the prediction
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            output = model(img_tensor)
            _, predicted_class_idx = torch.max(output, 1)
            predicted_class_idx = predicted_class_idx.item()

        # 4. Map the class index to the class name
        class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        predicted_class_name = class_names[predicted_class_idx]

        return predicted_class_name

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"


def process_and_predict(image_path):
    """
    Processes the image from the given path and predicts the class.
    This function handles both image path and PIL Image input.
    """
    try:
        if isinstance(image_path, str):
            # Load image from path
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
        elif isinstance(image_path, Image.Image):
            # If it's already a PIL Image, use it directly
            image = image_path
        else:
            raise ValueError("Input must be a file path or a PIL Image.")

        predicted_class = predict_image(image)
        return predicted_class
    except Exception as e:
        return f"Error: {e}"
    
def get_detailed_report(image_path):
    """
    Generates a detailed report based on the predicted tumor type.

    Args:
        image_path (str or PIL.Image.Image): Path to the MRI image or PIL Image.

    Returns:
        str: A detailed report including tumor type, potential surgery, and next steps.
    """
    predicted_class = process_and_predict(image_path)  # Use the process_and_predict function

    if predicted_class == "Error":
        return "Error: Unable to make a prediction. Please check the image and try again."

    report = f"Predicted Tumor Type: {predicted_class}\n\n"

    if predicted_class == "glioma":
        report += "Gliomas are tumors that start in the glial cells of the brain.  Surgery, radiation, and chemotherapy are common treatments.\n\n"
        report += "Potential Cognitive Surgery: Surgery aims to remove as much of the tumor as possible without damaging surrounding healthy brain tissue.  The impact on cognitive function depends on the tumor's location and the extent of surgery.  Rehabilitation may be needed.\n\n"
        report += "Immediate Next Steps:\n"
        report += "1. Consult with a neuro-oncologist for a comprehensive evaluation.\n"
        report += "2. Discuss treatment options, including surgery, radiation, and chemotherapy.\n"
        report += "3. Undergo further imaging (e.g., MRI) to precisely determine the tumor's extent.\n"
        report += "4.  Discuss potential cognitive impacts and rehabilitation options.\n"

    elif predicted_class == "meningioma":
        report += "Meningiomas are tumors that arise from the meninges, the membranes surrounding the brain and spinal cord. Many are benign and slow-growing.\n\n"
        report += "Potential Cognitive Surgery:  If surgery is needed, the goal is complete tumor removal.  Cognitive effects are usually minimal unless the tumor is in a critical area.  \n\n"
        report += "Immediate Next Steps:\n"
        report += "1. Consult with a neurologist or neurosurgeon.\n"
        report += "2. Determine the tumor's size, location, and growth rate with imaging.\n"
        report += "3. Discuss treatment options: observation, surgery, or radiosurgery.\n"
        report += "4.  Assess if the tumor is impacting cognitive function.\n"

    elif predicted_class == "no_tumor":
        report += "No tumor detected.  The image appears normal.\n\n"
        report += "Potential Cognitive Surgery:  Not applicable.\n\n"
        report += "Immediate Next Steps:\n"
        report += "1. Continue with regular checkups as recommended by your doctor.\n"
        report += "2.  Report any new or concerning symptoms to your doctor.\n"

    elif predicted_class == "pituitary":
        report += "Pituitary tumors are tumors that occur in the pituitary gland. They can cause hormonal imbalances.\n\n"
        report += "Potential Cognitive Surgery: Surgery, if needed, is often performed through the nose (transsphenoidal). Cognitive effects are usually minimal, but hormonal imbalances can indirectly affect cognition.\n\n"
        report += "Immediate Next Steps:\n"
        report += "1. Consult with an endocrinologist and a neurosurgeon.\n"
        report += "2.  Evaluate hormone levels and visual field testing.\n"
        report += "3.  Discuss treatment options, including medication, surgery, and radiation therapy.\n"
        report += "4.  Assess for any cognitive changes related to hormonal imbalances.\n"
    else:
        report += "Error: Unable to provide specific information."

    return report
    
def launch_gradio_interface():
    """
    Launches the Gradio interface for the brain tumor classifier.
    """
    # 1. Create the Gradio interface
    iface = gr.Interface(
        fn=get_detailed_report,  # Use the get_detailed_report function
        inputs=gr.Image(type="pil", label="Upload MRI Image"),
        outputs=gr.Text(label="Detailed Report"),  # Change the label
        title="Brain Tumor Classifier",
        description="Upload an MRI image to predict the type of brain tumor and get a detailed report.",
    )

    # 2. Launch the interface
    iface.launch()

if __name__ == "__main__":
    launch_gradio_interface()
