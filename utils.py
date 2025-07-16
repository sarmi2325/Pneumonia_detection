import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    try:
        # Step 1: Ensure model is built
        _ = model(img_array, training=False)

        # Step 2: Recursively search for the submodel that contains the target layer
        def find_layer_model(current_model, layer_name):
            # Check if current model contains the target layer
            if any(layer.name == layer_name for layer in current_model.layers):
                return current_model
            # Else, search within submodels
            for layer in current_model.layers:
                if isinstance(layer, tf.keras.Model):
                    try:
                        _ = layer(img_array, training=False)
                        if any(l.name == layer_name for l in layer.layers):
                            return layer
                    except:
                        pass
            raise ValueError(f"Layer '{layer_name}' not found in any submodel.")

        # Step 3: Get the submodel containing the last conv layer
        target_model = find_layer_model(model, last_conv_layer_name)
        _ = target_model(img_array, training=False)

        # Step 4: Define grad model
        grad_model = tf.keras.models.Model(
            inputs=[target_model.input],
            outputs=[target_model.get_layer(last_conv_layer_name).output, target_model.output]
        )

        # Step 5: Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            loss = predictions[:, 0]  # Assuming binary classification

        grads = tape.gradient(loss, conv_outputs)

        # Step 6: Compute weighted heatmap
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Step 7: Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    except Exception as e:
        raise RuntimeError(f"Grad-CAM failed for layer '{last_conv_layer_name}': {e}")

def superimpose_heatmap(heatmap, original_image, alpha=0.4):
    img = np.array(original_image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed



