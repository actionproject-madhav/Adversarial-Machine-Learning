"""
Iterative FGSM Attack - More powerful than single-step FGSM
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("ITERATIVE FGSM (I-FGSM) ATTACK")
print("="*80)

# Load model
model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
model.trainable = False

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, 0)
    return image

# Load image
image_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)
image = preprocess(image)

# Get original prediction
orig_probs = model.predict(image)
orig_class_idx = np.argmax(orig_probs)
_, orig_class, orig_conf = decode_predictions(orig_probs, top=1)[0][0]
print(f"Original: {orig_class} ({orig_conf*100:.1f}%)")

def iterative_fgsm(model, image, eps, alpha, num_iter, targeted=False, target_class=None):
    """
    Iterative FGSM Attack
    
    Args:
        model: Target model
        image: Input image
        eps: Maximum perturbation
        alpha: Step size
        num_iter: Number of iterations
        targeted: If True, perform targeted attack
        target_class: Target class for targeted attack
    """
    adv_image = tf.identity(image)
    
    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            
            if targeted:
                # Minimize loss for target class
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    [target_class], prediction
                )
            else:
                # Maximize loss for true class
                true_class = np.argmax(model.predict(image, verbose=0))
                loss = -tf.keras.losses.sparse_categorical_crossentropy(
                    [true_class], prediction
                )
        
        # Get gradients
        gradient = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradient)
        
        # Update adversarial image
        if targeted:
            adv_image = adv_image - alpha * signed_grad
        else:
            adv_image = adv_image + alpha * signed_grad
        
        # Clip to maintain L-infinity constraint
        perturbation = tf.clip_by_value(adv_image - image, -eps, eps)
        adv_image = image + perturbation
        
        # Clip to valid range
        adv_image = tf.clip_by_value(adv_image, -1, 1)
    
    return adv_image

print("\n" + "="*80)
print("TESTING I-FGSM WITH DIFFERENT PARAMETERS")
print("="*80)

# Test different configurations
configs = [
    {"eps": 0.03, "alpha": 0.01, "num_iter": 10},
    {"eps": 0.05, "alpha": 0.01, "num_iter": 20},
    {"eps": 0.07, "alpha": 0.005, "num_iter": 40},
    {"eps": 0.1, "alpha": 0.01, "num_iter": 30},
]

results = []
for config in configs:
    print(f"\nConfig: eps={config['eps']}, alpha={config['alpha']}, iterations={config['num_iter']}")
    print("-" * 40)
    
    # Untargeted attack
    adv_image = iterative_fgsm(
        model, image, 
        eps=config['eps'], 
        alpha=config['alpha'], 
        num_iter=config['num_iter'],
        targeted=False
    )
    
    adv_probs = model.predict(adv_image, verbose=0)
    _, adv_class, adv_conf = decode_predictions(adv_probs, top=1)[0][0]
    
    success = adv_class != orig_class
    print(f"Untargeted: {adv_class} ({adv_conf*100:.1f}%) - {'✓ SUCCESS' if success else '✗ Failed'}")
    
    # Targeted attack to goldfish (class 1)
    target_class = 1  # goldfish
    adv_image_targeted = iterative_fgsm(
        model, image,
        eps=config['eps'],
        alpha=config['alpha'],
        num_iter=config['num_iter'],
        targeted=True,
        target_class=target_class
    )
    
    adv_probs_t = model.predict(adv_image_targeted, verbose=0)
    _, adv_class_t, adv_conf_t = decode_predictions(adv_probs_t, top=1)[0][0]
    
    success_t = adv_class_t == 'goldfish'
    print(f"Targeted (goldfish): {adv_class_t} ({adv_conf_t*100:.1f}%) - {'✓ SUCCESS' if success_t else '✗ Failed'}")
    
    results.append({
        'config': config,
        'untargeted': (adv_image, adv_class, adv_conf, success),
        'targeted': (adv_image_targeted, adv_class_t, adv_conf_t, success_t)
    })

# Find best result
best_untargeted = max(results, key=lambda x: x['untargeted'][3])
best_targeted = max(results, key=lambda x: x['targeted'][3])

# Visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Original
for i in range(2):
    axes[i, 0].imshow(image[0] * 0.5 + 0.5)
    axes[i, 0].set_title(f'Original\n{orig_class}\n{orig_conf*100:.1f}%', fontsize=11)
    axes[i, 0].axis('off')

# Untargeted results
config = best_untargeted['config']
adv_img, adv_cls, adv_cnf, success = best_untargeted['untargeted']

axes[0, 1].imshow(adv_img[0] * 0.5 + 0.5)
axes[0, 1].set_title(f'I-FGSM Untargeted\n{adv_cls}\n{adv_cnf*100:.1f}%', 
                     fontsize=11, color='green' if success else 'red')
axes[0, 1].axis('off')

# Show perturbation
perturbation = adv_img - image
axes[0, 2].imshow(perturbation[0] * 0.5 + 0.5)
axes[0, 2].set_title(f'Perturbation\neps={config["eps"]}', fontsize=11)
axes[0, 2].axis('off')

# Difference map
diff = tf.abs(adv_img - image)
axes[0, 3].imshow(tf.reduce_mean(diff, axis=-1)[0], cmap='hot')
axes[0, 3].set_title('Difference Heatmap', fontsize=11)
axes[0, 3].axis('off')

# Targeted results
config_t = best_targeted['config']
adv_img_t, adv_cls_t, adv_cnf_t, success_t = best_targeted['targeted']

axes[1, 1].imshow(adv_img_t[0] * 0.5 + 0.5)
axes[1, 1].set_title(f'I-FGSM Targeted\n{adv_cls_t}\n{adv_cnf_t*100:.1f}%', 
                     fontsize=11, color='green' if success_t else 'orange')
axes[1, 1].axis('off')

# Show perturbation
perturbation_t = adv_img_t - image
axes[1, 2].imshow(perturbation_t[0] * 0.5 + 0.5)
axes[1, 2].set_title(f'Perturbation\neps={config_t["eps"]}', fontsize=11)
axes[1, 2].axis('off')

# Difference map
diff_t = tf.abs(adv_img_t - image)
axes[1, 3].imshow(tf.reduce_mean(diff_t, axis=-1)[0], cmap='hot')
axes[1, 3].set_title('Difference Heatmap', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Iterative FGSM Attack Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ifgsm_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ATTACK SUMMARY")
print("="*80)
print(f"Original: {orig_class} ({orig_conf*100:.1f}%)")
print(f"Best Untargeted: {best_untargeted['untargeted'][1]} - {'✓' if best_untargeted['untargeted'][3] else '✗'}")
print(f"Best Targeted: {best_targeted['targeted'][1]} - {'✓' if best_targeted['targeted'][3] else '✗'}")
print("="*80)