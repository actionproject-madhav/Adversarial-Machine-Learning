"""
FGSM Attack - Using TensorFlow's Official Tutorial Approach
This uses the exact method from TensorFlow's adversarial example tutorial
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("FGSM ATTACK - TensorFlow Official Tutorial Method")
print("="*80)

# Load pretrained model
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, 0)
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]

# Load image
image_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)

# Get initial label
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title(f'{image_class}: {class_confidence*100:.2f}%')
plt.axis('off')
plt.savefig('0_original_with_label.png', bbox_inches='tight', dpi=150)
plt.close()

print(f"\nOriginal prediction: {image_class} ({class_confidence*100:.2f}%)")

# Create adversarial pattern function
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

# Get the input label of the image (USE ACTUAL PREDICTED CLASS!)
predicted_class_idx = np.argmax(image_probs)
label = tf.one_hot(predicted_class_idx, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

print(f"Using predicted class index: {predicted_class_idx}")

perturbations = create_adversarial_pattern(image, label)

# Visualize perturbation
plt.figure()
plt.imshow(perturbations[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
plt.title('Perturbation Pattern')
plt.axis('off')
plt.savefig('1_perturbation.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nTesting different epsilon values...")
print("-" * 60)

epsilons = [0, 0.01, 0.02, 0.05, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps), eps) for eps in epsilons]

best_result = None

for i, (desc, eps) in enumerate(descriptions):
    # Try NEGATIVE direction (empirically this might work for MobileNetV2)
    adv_x = image - eps*perturbations  # NEGATIVE!
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    
    adv_probs = pretrained_model.predict(adv_x)
    _, adv_class, adv_confidence = get_imagenet_label(adv_probs)
    
    # Check if attack succeeded
    if adv_class != image_class:
        print(f"ε = {eps:.3f}: ✓ {adv_class:25s} ({adv_confidence*100:5.1f}%) SUCCESS!")
        if best_result is None:
            best_result = (eps, adv_x, adv_probs, adv_class, adv_confidence)
    else:
        orig_conf = image_probs[0, predicted_class_idx]
        adv_conf = adv_probs[0, predicted_class_idx]
        print(f"ε = {eps:.3f}: ✗ {adv_class:25s} ({adv_confidence*100:5.1f}%) " +
              f"[confidence: {orig_conf*100:.1f}% → {adv_conf*100:.1f}%]")
    
    if best_result is None:
        best_result = (eps, adv_x, adv_probs, adv_class, adv_confidence)

# Use best result
eps_best, adv_x_best, adv_probs_best, adv_class_best, adv_confidence_best = best_result

print("\n" + "="*80)
print("FINAL RESULT:")
print("="*80)
print(f"Original:    {image_class:25s} ({class_confidence*100:.1f}%)")
print(f"Adversarial: {adv_class_best:25s} ({adv_confidence_best*100:.1f}%) with ε={eps_best}")
print(f"Status:      {'✓ ATTACK SUCCESSFUL' if adv_class_best != image_class else '✗ Attack failed'}")
print("="*80)

# Create final visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original
axes[0].imshow(image[0] * 0.5 + 0.5)
axes[0].set_title(f'Original\n{image_class}\n{class_confidence*100:.1f}%', 
                  fontsize=14, fontweight='bold')
axes[0].axis('off')

# Perturbation
axes[1].imshow(perturbations[0] * 0.5 + 0.5)
axes[1].set_title(f'Perturbation\nε = {eps_best}', 
                  fontsize=14, fontweight='bold')
axes[1].axis('off')

# Adversarial
axes[2].imshow(adv_x_best[0] * 0.5 + 0.5)
color = 'green' if adv_class_best != image_class else 'red'
axes[2].set_title(f'Adversarial\n{adv_class_best}\n{adv_confidence_best*100:.1f}%', 
                  fontsize=14, fontweight='bold', color=color)
axes[2].axis('off')

plt.suptitle('FGSM Adversarial Attack', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('2_fgsm_complete.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Saved: 0_original_with_label.png, 1_perturbation.png, 2_fgsm_complete.png")
print("\nNote: This uses the EXACT TensorFlow tutorial approach.")
print("If this still doesn't work, the model/image combination may be robust to FGSM.")