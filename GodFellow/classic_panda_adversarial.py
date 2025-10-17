import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("FGSM ATTACK - Correct Implementation")
print("="*80)

# Load model
print("\n[1/5] Loading MobileNetV2...")
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
print("✓ Model loaded")

# Load and preprocess image
print("[2/5] Loading image...")
# Use a different image - download a panda image (classic FGSM demo)
image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
image_path = tf.keras.utils.get_file('dog.jpg', image_url)
image_raw = tf.io.read_file(image_path)
image_decoded = tf.image.decode_jpeg(image_raw, channels=3)

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

image = preprocess(image_decoded)
image = tf.expand_dims(image, 0)
print("✓ Image preprocessed")

# Get initial prediction
print("[3/5] Getting initial prediction...")
image_probs = pretrained_model.predict(image, verbose=0)
initial_pred = decode_predictions(image_probs, top=1)[0][0]
true_class = int(np.argmax(image_probs))

print(f"✓ Prediction: {initial_pred[1]} ({initial_pred[2]*100:.1f}%)")
print(f"  True class index: {true_class}")
print(f"  True class prob: {image_probs[0, true_class]:.4f}")

# CORRECT FGSM: Compute gradient of cross-entropy loss
print("[4/5] Computing FGSM perturbation...")
image_variable = tf.Variable(image)

with tf.GradientTape() as tape:
    tape.watch(image_variable)
    preds = pretrained_model(image_variable, training=False)  # probabilities
    # Cross-entropy loss (higher loss = more wrong)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=tf.constant([true_class]),  # Convert to tensor
        y_pred=preds, 
        from_logits=False  # preds are probabilities, not logits
    )
    loss = tf.reduce_mean(loss)  # scalar

grad = tape.gradient(loss, image_variable)
# FGSM uses NEGATIVE gradient to minimize the true class score
perturbation = -tf.sign(grad)  # NEGATIVE!

# Debug info
print(f"  Loss value: {loss.numpy():.4f}")
print(f"  Gradient L-inf norm: {tf.reduce_max(tf.abs(grad)).numpy():.6f}")
print(f"  Non-zero perturbations: {np.sum(perturbation.numpy() != 0):,}")

# Create adversarial example - TARGETED attack
print("[5/5] Creating adversarial image...")
eps = 0.15  # Larger epsilon to fully change prediction

# Targeted FGSM: x_adv = x - ε × sign(∇L_target)
adv_x = image_variable + eps * perturbation
adv_x = tf.clip_by_value(adv_x, -1.0, 1.0)

# Evaluate
adv_probs = pretrained_model.predict(adv_x, verbose=0)
adv_pred = decode_predictions(adv_probs, top=1)[0][0]
adv_class = int(np.argmax(adv_probs))

success = adv_class == target_class
print(f"\n{'='*80}")
print(f"RESULT:")
print(f"{'='*80}")
print(f"Original: {initial_pred[1]:20s} ({initial_pred[2]*100:.1f}%)")
print(f"Target:   goldfish")
print(f"Adversarial: {adv_pred[1]:20s} ({adv_pred[2]*100:.1f}%)")
print(f"\nTarget class confidence:")
print(f"  Before: {image_probs[0, target_class]*100:.2f}%")
print(f"  After:  {adv_probs[0, target_class]*100:.2f}%")
print(f"\nAttack {'✓ SUCCESSFUL' if success else '✗ FAILED (but may have changed prediction)'}")

# Show mathematical example
print(f"\n{'='*80}")
print("MATHEMATICAL EXAMPLE (5×5 patch):")
print(f"{'='*80}")

patch_orig = image_variable[0, :5, :5, 0].numpy()
patch_grad = grad[0, :5, :5, 0].numpy()
patch_sign = perturbation[0, :5, :5, 0].numpy()
patch_adv = adv_x[0, :5, :5, 0].numpy()

print(f"\n1. Original pixels:")
print(patch_orig)

print(f"\n2. Gradient of cross-entropy loss:")
print(patch_grad)

print(f"\n3. Sign of gradient:")
print(patch_sign.astype(int))

print(f"\n4. FGSM formula: x_adv = x - ε × sign(∇L)")
print(f"   where ε = {eps}")
print(f"\n   Adversarial pixels:")
print(patch_adv)

print(f"\n5. Difference (perturbation added):")
print(patch_adv - patch_orig)

print(f"\n{'='*80}")
print("KEY INSIGHT:")
print(f"{'='*80}")
print("• Cross-entropy loss measures how WRONG the prediction is")
print("• Gradient ∇L shows direction that INCREASES loss")
print("• We use NEGATIVE gradient to DECREASE true class confidence")
print("• FGSM: x_adv = x - ε × sign(∇L)")
print(f"{'='*80}")

# Create visualizations
print("\nCreating images...")

# IMAGE 1: Original
fig = plt.figure(figsize=(8, 8))
original_display = (image_variable[0].numpy() - image_variable[0].numpy().min()) / \
                   (image_variable[0].numpy().max() - image_variable[0].numpy().min())
plt.imshow(original_display)
plt.title(f'ORIGINAL\n{initial_pred[1]}\n{initial_pred[2]*100:.1f}%',
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('1_original.png', dpi=150, bbox_inches='tight')
plt.close()

# IMAGE 2: Perturbation
fig = plt.figure(figsize=(8, 8))
pert_vis = perturbation[0].numpy() * 0.5 + 0.5
plt.imshow(pert_vis)
plt.title(f'PERTURBATION\nsign(∇L)\nε = {eps}',
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('2_perturbation.png', dpi=150, bbox_inches='tight')
plt.close()

# IMAGE 3: Adversarial
fig = plt.figure(figsize=(8, 8))
adv_display = (adv_x[0].numpy() - adv_x[0].numpy().min()) / \
              (adv_x[0].numpy().max() - adv_x[0].numpy().min())
plt.imshow(adv_display)
color = 'green' if success else 'orange'
plt.title(f'ADVERSARIAL (ε={eps})\n{adv_pred[1]}\n{adv_pred[2]*100:.1f}%',
          fontsize=18, fontweight='bold', pad=20, color=color)
plt.axis('off')
plt.tight_layout()
plt.savefig('3_adversarial.png', dpi=150, bbox_inches='tight')
plt.close()

# IMAGE 4: Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(original_display)
axes[0].set_title(f'BEFORE\n{initial_pred[1]}\n{initial_pred[2]*100:.1f}%',
                  fontsize=16, fontweight='bold', pad=20)
axes[0].axis('off')
axes[1].imshow(adv_display)
axes[1].set_title(f'AFTER\n{adv_pred[1]}\n{adv_pred[2]*100:.1f}%',
                  fontsize=16, fontweight='bold', pad=20, color=color)
axes[1].axis('off')
plt.suptitle(f'FGSM Attack (ε={eps})', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('4_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: 1_original.png, 2_perturbation.png, 3_adversarial.png, 4_comparison.png")

# If attack didn't fully succeed but confidence dropped
if not success and image_probs[0, true_class] > adv_probs[0, true_class]:
    print(f"\nNote: While top-1 class didn't change, confidence dropped by")
    print(f"      {(image_probs[0, true_class] - adv_probs[0, true_class])*100:.2f}%!")
    print(f"      Try larger epsilon (e.g., 0.05-0.1) for successful misclassification.")

print("\nDone!")