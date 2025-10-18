import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Set matplotlib parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = False

print("TensorFlow version:", tf.__version__)

# Load pretrained MobileNetV2 model
print("Loading MobileNetV2 model...")
pretrained_model = tf.keras.applications.MobileNetV2(
    include_top=True,
    weights='imagenet'
)
pretrained_model.trainable = False

# ImageNet labels decoder
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

print("Model loaded successfully!")

# Helper function to preprocess the image
def preprocess(image):
    """Preprocess image for MobileNetV2"""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
    """Extract top prediction label"""
    return decode_predictions(probs, top=1)[0][0]

# Load and preprocess the image
print("\nDownloading sample image...")
image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
response = requests.get(image_url)
image_raw = Image.open(BytesIO(response.content))
image_raw = np.array(image_raw)

# Preprocess image
image = preprocess(image_raw)
image_probs = pretrained_model.predict(image, verbose=0)

# Display original image
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # Convert from [-1, 1] to [0, 1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title(f'Original: {image_class} ({class_confidence*100:.2f}% confidence)')
plt.axis('off')
plt.tight_layout()
plt.savefig('original_image.png', dpi=150, bbox_inches='tight')
print(f"\nOriginal prediction: {image_class} with {class_confidence*100:.2f}% confidence")
plt.show()

# Create adversarial pattern using FGSM
print("\nCreating adversarial pattern...")

loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    """
    Generate adversarial perturbation using Fast Gradient Sign Method (FGSM)
    """
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    
    # Get gradients of loss w.r.t input image
    gradient = tape.gradient(loss, input_image)
    # Get sign of gradients to create perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

# Get the correct label for the image (Labrador Retriever)
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

# Generate perturbations
perturbations = create_adversarial_pattern(image, label)

# Visualize perturbations
plt.figure()
plt.imshow(perturbations[0] * 0.5 + 0.5)  # Convert from [-1, 1] to [0, 1]
plt.title('Adversarial Perturbations Pattern')
plt.axis('off')
plt.tight_layout()
plt.savefig('perturbations.png', dpi=150, bbox_inches='tight')
print("Perturbations generated successfully!")
plt.show()

# Test different epsilon values
print("\nTesting different epsilon values...")

def display_adversarial_images(image, perturbations, epsilons):
    """Display adversarial images for different epsilon values"""
    n = len(epsilons)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    for i, eps in enumerate(epsilons):
        # Create adversarial image
        adv_x = image + eps * perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        
        # Get prediction
        adv_probs = pretrained_model.predict(adv_x, verbose=0)
        _, label, confidence = get_imagenet_label(adv_probs)
        
        # Display
        if n > 1:
            ax = axes[i]
        else:
            ax = axes
            
        ax.imshow(adv_x[0] * 0.5 + 0.5)
        
        if eps == 0:
            title = f'Original\n{label}\n{confidence*100:.2f}% confidence'
        else:
            title = f'Epsilon = {eps}\n{label}\n{confidence*100:.2f}% confidence'
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        print(f"Epsilon {eps}: {label} ({confidence*100:.2f}% confidence)")
    
    plt.tight_layout()
    plt.savefig('adversarial_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

# Test with different epsilon values
epsilons = [0, 0.01, 0.1, 0.15]
display_adversarial_images(image, perturbations, epsilons)

print("\n" + "="*60)
print("FGSM Attack Complete!")
print("="*60)
print("\nKey observations:")
print("- Epsilon = 0: Original image, correct classification")
print("- Epsilon = 0.01: Small perturbation, may still classify correctly")
print("- Epsilon = 0.1: Moderate perturbation, likely misclassification")
print("- Epsilon = 0.15: Strong perturbation, definite misclassification")
print("\nAs epsilon increases:")
print("  ✓ Attack success rate increases")
print("  ✗ Perturbations become more visible")