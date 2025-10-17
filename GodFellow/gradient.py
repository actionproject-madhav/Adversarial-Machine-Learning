import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("FGSM ATTACK - Simple")
print("="*80)

# Load model
print("\n[1/5] Loading MobileNetV2...")
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
print("✓ Done")

# Load image
print("[2/5] Loading image...")
image_path = tf.keras.utils.get_file('grace_hopper.jpg', 
                                     'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
image_raw = tf.io.read_file(image_path)
image_decoded = tf.image.decode_jpeg(image_raw, channels=3)

# Preprocess
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

image = preprocess(image_decoded)
image = tf.expand_dims(image, 0)
print("✓ Done")

# Get initial prediction
print("[3/5] Getting prediction...")
image_probs = pretrained_model.predict(image, verbose=0)
initial_pred = decode_predictions(image_probs, top=1)[0][0]
print(f"✓ Predicts: {initial_pred[1]} ({initial_pred[2]*100:.1f}%)")

# Create adversarial perturbation
print("[4/5] Computing gradient...")
image_variable = tf.Variable(image)
true_class = np.argmax(image_probs)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

with tf.GradientTape() as tape:
    tape.watch(image_variable)
    prediction = pretrained_model(image_variable, training=False)
    loss = loss_object(tf.constant([true_class]), prediction)

gradient = tape.gradient(loss, image_variable)
perturbations = tf.sign(gradient)
print(f"✓ Gradient computed (Loss: {loss.numpy():.4f})")

# Create adversarial example with larger epsilon
print("[5/5] Creating adversarial image...")
eps = 0.07  # Larger epsilon to ensure attack works
adv_x = image_variable + eps * perturbations
adv_x = tf.clip_by_value(adv_x, -1, 1)

adv_probs = pretrained_model.predict(adv_x, verbose=0)
adv_pred = decode_predictions(adv_probs, top=1)[0][0]

success = adv_pred[1] != initial_pred[1]
print(f"✓ Attack {'SUCCESSFUL' if success else 'FAILED'}")
print(f"  Now predicts: {adv_pred[1]} ({adv_pred[2]*100:.1f}%)")

# Show exact math on small patch
print("\n" + "="*80)
print("HOW IT WORKS (5×5 patch example):")
print("="*80)
patch_orig = image_variable[0, :5, :5, 0].numpy()
patch_grad = gradient[0, :5, :5, 0].numpy()
patch_sign = perturbations[0, :5, :5, 0].numpy()
patch_adv = adv_x[0, :5, :5, 0].numpy()

print(f"\n1. Original pixels:")
print(patch_orig)

print(f"\n2. Gradient (direction to INCREASE loss for 'military_uniform'):")
print(patch_grad)

print(f"\n3. Sign of gradient (-1, 0, or +1):")
print(patch_sign.astype(int))

print(f"\n4. Add ε={eps} × sign:")
print(f"   Adversarial = Original + {eps} × sign(gradient)")
print(patch_adv)

print(f"\n5. Difference:")
print(patch_adv - patch_orig)

print("\n" + "="*80)
print("KEY CONCEPT:")
print("="*80)
print("• Gradient shows direction that INCREASES loss for correct class")
print("• Adding ε × sign(gradient) makes model LESS confident in 'military_uniform'")
print("• Model is forced to predict something else!")
print("="*80)

# Create 4 simple images
print("\nCreating images...")
# IMAGE 1
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

# IMAGE 2
fig = plt.figure(figsize=(8, 8))
pert_vis = perturbations[0].numpy() * 0.5 + 0.5
plt.imshow(pert_vis)
plt.title(f'PERTURBATION\nsign(gradient)',
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('2_perturbation.png', dpi=150, bbox_inches='tight')
plt.close()

# IMAGE 3
fig = plt.figure(figsize=(8, 8))
adv_display = (adv_x[0].numpy() - adv_x[0].numpy().min()) / \
              (adv_x[0].numpy().max() - adv_x[0].numpy().min())
plt.imshow(adv_display)
plt.title(f'ADVERSARIAL (ε={eps})\n{adv_pred[1]}\n{adv_pred[2]*100:.1f}%',
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('3_adversarial.png', dpi=150, bbox_inches='tight')
plt.close()

# IMAGE 4
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(original_display)
axes[0].set_title(f'BEFORE\n{initial_pred[1]}\n{initial_pred[2]*100:.1f}%',
                  fontsize=16, fontweight='bold', pad=20)
axes[0].axis('off')
axes[1].imshow(adv_display)
axes[1].set_title(f'AFTER\n{adv_pred[1]}\n{adv_pred[2]*100:.1f}%',
                  fontsize=16, fontweight='bold', pad=20)
axes[1].axis('off')
plt.suptitle(f'FGSM Attack (ε={eps})', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('4_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: 1_original.png, 2_perturbation.png, 3_adversarial.png, 4_comparison.png")
print("\nDone!")