import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("FGSM ATTACK - Working Implementation")
print("="*80)

# Load model
print("\n[1/4] Loading MobileNetV2...")
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
print("✓ Model loaded")

# Load image
print("[2/4] Loading dog image...")
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
print("[3/4] Getting initial prediction...")
image_probs = pretrained_model.predict(image, verbose=0)
initial_pred = decode_predictions(image_probs, top=1)[0][0]
true_class = int(np.argmax(image_probs))
print(f"✓ Predicts: {initial_pred[1]} ({initial_pred[2]*100:.1f}%)")

# UNTARGETED FGSM Attack - just make it predict something wrong
print("[4/4] Running UNTARGETED FGSM attack...")
print(f"  Goal: Make model predict ANYTHING except {initial_pred[1]}")

image_variable = tf.Variable(image)

with tf.GradientTape() as tape:
    tape.watch(image_variable)
    preds = pretrained_model(image_variable, training=False)
    # Loss for TRUE class (we want to MAXIMIZE this loss)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=tf.constant([true_class]),
        y_pred=preds, 
        from_logits=False
    )
    loss = tf.reduce_mean(loss)

grad = tape.gradient(loss, image_variable)

# For UNtargeted attack: maximize loss for true class
# Gradient points in direction of INCREASING loss
# So we ADD positive gradient
perturbation = tf.sign(grad)

# Try multiple epsilons
epsilons = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
print(f"\n  Testing different epsilon values...")

best_eps = None
best_adv = None
best_probs = None

for eps in epsilons:
    adv_x = image_variable + eps * perturbation
    adv_x = tf.clip_by_value(adv_x, -1.0, 1.0)
    
    adv_probs = pretrained_model.predict(adv_x, verbose=0)
    adv_pred = decode_predictions(adv_probs, top=1)[0][0]
    adv_class = int(np.argmax(adv_probs))
    
    true_conf = adv_probs[0, true_class]
    
    success = adv_class != true_class
    
    print(f"    ε={eps:.2f}: {adv_pred[1]:20s} ({adv_pred[2]*100:5.1f}%) " + 
          f"| True class: {true_conf*100:5.1f}% {'✓ FOOLED!' if success else ''}")
    
    if success and best_eps is None:
        best_eps = eps
        best_adv = adv_x
        best_probs = adv_probs
        break
    
    if best_eps is None:
        best_eps = eps
        best_adv = adv_x
        best_probs = adv_probs

# Use best result
adv_pred = decode_predictions(best_probs, top=1)[0][0]
adv_class = int(np.argmax(best_probs))
success = adv_class == target_class

print(f"\n{'='*80}")
print("RESULT:")
print(f"{'='*80}")
print(f"Original:    {initial_pred[1]:20s} ({initial_pred[2]*100:.1f}%)")
print(f"Adversarial: {adv_pred[1]:20s} ({adv_pred[2]*100:.1f}%) with ε={best_eps}")
print(f"Original class confidence: {image_probs[0, true_class]*100:.1f}% → {best_probs[0, true_class]*100:.1f}%")
print(f"\nAttack {'✓ SUCCESSFUL!' if success else '✗ Failed'}")
print(f"{'='*80}")

# Create visualizations
print("\nCreating images...")

# Original
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

# Perturbation
fig = plt.figure(figsize=(8, 8))
pert_vis = perturbation[0].numpy() * 0.5 + 0.5
plt.imshow(pert_vis)
plt.title(f'PERTURBATION\nUntargeted FGSM\nε = {best_eps}',
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('2_perturbation.png', dpi=150, bbox_inches='tight')
plt.close()

# Adversarial
fig = plt.figure(figsize=(8, 8))
adv_display = (best_adv[0].numpy() - best_adv[0].numpy().min()) / \
              (best_adv[0].numpy().max() - best_adv[0].numpy().min())
plt.imshow(adv_display)
color = 'green' if success else 'orange'
plt.title(f'ADVERSARIAL\n{adv_pred[1]}\n{adv_pred[2]*100:.1f}%',
          fontsize=18, fontweight='bold', pad=20, color=color)
plt.axis('off')
plt.tight_layout()
plt.savefig('3_adversarial.png', dpi=150, bbox_inches='tight')
plt.close()

# Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(original_display)
axes[0].set_title(f'BEFORE\n{initial_pred[1]}\n{initial_pred[2]*100:.1f}%',
                  fontsize=16, fontweight='bold', pad=20)
axes[0].axis('off')
axes[1].imshow(adv_display)
axes[1].set_title(f'AFTER (ε={best_eps})\n{adv_pred[1]}\n{adv_pred[2]*100:.1f}%',
                  fontsize=16, fontweight='bold', pad=20, color=color)
axes[1].axis('off')
plt.suptitle('Untargeted FGSM Attack', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('4_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: 1_original.png, 2_perturbation.png, 3_adversarial.png, 4_comparison.png")
print("\nDone!")