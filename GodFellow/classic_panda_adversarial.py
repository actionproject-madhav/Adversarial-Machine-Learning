"""
Improved FGSM Attack Implementation
This version implements both targeted and untargeted attacks correctly
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("IMPROVED FGSM ATTACK")
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
plt.figure(figsize=(8, 8))
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title(f'{image_class}: {class_confidence*100:.2f}%')
plt.axis('off')
plt.savefig('0_original.png', bbox_inches='tight', dpi=150)
plt.close()

print(f"\nOriginal prediction: {image_class} ({class_confidence*100:.2f}%)")
predicted_class_idx = np.argmax(image_probs)

# Method 1: Untargeted Attack (maximize loss for true class)
def create_untargeted_pattern(input_image, true_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        # Use negative log likelihood of true class
        loss = -tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(true_class_idx, 1000),
            logits=prediction
        )
    
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Method 2: Targeted Attack (minimize loss for target class)
def create_targeted_pattern(input_image, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        # Maximize probability of target class
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(target_class_idx, 1000),
            logits=prediction
        )
    
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

print("\n" + "="*80)
print("UNTARGETED ATTACK - Trying to change prediction from Labrador")
print("="*80)

# Create untargeted perturbation
perturbations_untargeted = create_untargeted_pattern(image, [predicted_class_idx])

# Test different epsilon values
epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
print("\nTesting epsilon values...")
print("-" * 60)

best_untargeted = None
for eps in epsilons:
    # Apply perturbation in POSITIVE direction for untargeted attack
    adv_x = image + eps * perturbations_untargeted
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    
    adv_probs = pretrained_model.predict(adv_x, verbose=0)
    _, adv_class, adv_confidence = get_imagenet_label(adv_probs)
    
    if adv_class != image_class:
        print(f"ε = {eps:.3f}: ✓ SUCCESS! Changed to: {adv_class} ({adv_confidence*100:.1f}%)")
        if best_untargeted is None:
            best_untargeted = (eps, adv_x, adv_class, adv_confidence)
    else:
        orig_conf = image_probs[0, predicted_class_idx]
        adv_conf = adv_probs[0, predicted_class_idx]
        print(f"ε = {eps:.3f}: Still {adv_class} (confidence: {orig_conf*100:.1f}% → {adv_conf*100:.1f}%)")

print("\n" + "="*80)
print("TARGETED ATTACK - Trying to make it predict 'tennis_ball' (class 852)")
print("="*80)

# Targeted attack to tennis ball (class 852)
target_class = 852  # tennis_ball
perturbations_targeted = create_targeted_pattern(image, [target_class])

print("\nTesting epsilon values for targeted attack...")
print("-" * 60)

best_targeted = None
for eps in epsilons:
    # Apply perturbation in NEGATIVE direction for targeted attack
    adv_x = image - eps * perturbations_targeted
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    
    adv_probs = pretrained_model.predict(adv_x, verbose=0)
    _, adv_class, adv_confidence = get_imagenet_label(adv_probs)
    target_conf = adv_probs[0, target_class]
    
    if adv_class == 'tennis_ball':
        print(f"ε = {eps:.3f}: ✓ SUCCESS! Predicted tennis_ball ({adv_confidence*100:.1f}%)")
        if best_targeted is None:
            best_targeted = (eps, adv_x, adv_class, adv_confidence)
    else:
        print(f"ε = {eps:.3f}: {adv_class} ({adv_confidence*100:.1f}%), tennis_ball conf: {target_conf*100:.2f}%")

# Create final visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Untargeted Attack
axes[0, 0].imshow(image[0] * 0.5 + 0.5)
axes[0, 0].set_title(f'Original\n{image_class}\n{class_confidence*100:.1f}%', fontsize=12)
axes[0, 0].axis('off')

if best_untargeted:
    eps_u, adv_x_u, adv_class_u, adv_conf_u = best_untargeted
    axes[0, 1].imshow(perturbations_untargeted[0] * 0.5 + 0.5)
    axes[0, 1].set_title(f'Untargeted Perturbation\nε = {eps_u}', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(adv_x_u[0] * 0.5 + 0.5)
    axes[0, 2].set_title(f'Result: {adv_class_u}\n{adv_conf_u*100:.1f}%', 
                        fontsize=12, color='green')
    axes[0, 2].axis('off')
else:
    axes[0, 1].text(0.5, 0.5, 'Attack Failed', ha='center', va='center', fontsize=14)
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

# Row 2: Targeted Attack
axes[1, 0].imshow(image[0] * 0.5 + 0.5)
axes[1, 0].set_title(f'Original\n{image_class}\n{class_confidence*100:.1f}%', fontsize=12)
axes[1, 0].axis('off')

if best_targeted:
    eps_t, adv_x_t, adv_class_t, adv_conf_t = best_targeted
    axes[1, 1].imshow(perturbations_targeted[0] * 0.5 + 0.5)
    axes[1, 1].set_title(f'Targeted Perturbation\n(to tennis_ball, ε = {eps_t})', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(adv_x_t[0] * 0.5 + 0.5)
    axes[1, 2].set_title(f'Result: {adv_class_t}\n{adv_conf_t*100:.1f}%', 
                        fontsize=12, color='green' if adv_class_t == 'tennis_ball' else 'orange')
    axes[1, 2].axis('off')
else:
    axes[1, 1].text(0.5, 0.5, 'Attack Failed', ha='center', va='center', fontsize=14)
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')

plt.suptitle('FGSM Attack Results: Untargeted vs Targeted', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fgsm_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Original: {image_class} ({class_confidence*100:.1f}%)")
if best_untargeted:
    print(f"Untargeted Attack: ✓ Changed to {best_untargeted[2]} with ε={best_untargeted[0]}")
else:
    print("Untargeted Attack: ✗ Failed")
if best_targeted:
    print(f"Targeted Attack: ✓ Changed to {best_targeted[2]} with ε={best_targeted[0]}")
else:
    print("Targeted Attack: ✗ Failed")
print("="*80)