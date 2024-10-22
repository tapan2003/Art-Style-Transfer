import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_and_preprocess_image(image_path, max_dim=512):
    img = load_img(image_path)
    img = img_to_array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = np.expand_dims(img, axis=0).copy()  # Ensure the array is writable
    img = tf.keras.applications.vgg19.preprocess_input(
        img)  # VGG19 preprocessing
    return img


def deprocess_image(image):
    # Convert tensor to numpy array and remove batch dimension
    image = image.numpy().squeeze()
    image = image[:, :, ::-1]  # Convert from BGR to RGB
    image += [103.939, 116.779, 123.68]  # VGG19 mean pixel values
    return np.clip(image, 0, 255).astype('uint8')


def save_image(image, path):
    img = deprocess_image(image.numpy())
    img = Image.fromarray(img)
    img.save(path)


# Define layers of interest
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
content_layers = ['block5_conv2']
output_layers = style_layers + content_layers
NUM_CONTENT_LAYERS = len(content_layers)
NUM_STYLE_LAYERS = len(style_layers)


def vgg_model(layer_names):
    """Creates a VGG model that outputs the style and content layer activations."""
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)


def gram_matrix(input_tensor):
    """Calculates the gram matrix."""
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[
                            1] * tf.shape(input_tensor)[2], tf.float32)
    return gram / num_locations


def get_style_image_features(image, model):
    """Get the style image features."""
    outputs = model(image)
    return [gram_matrix(style_layer) for style_layer in outputs[:NUM_STYLE_LAYERS]]


def get_content_image_features(image, model):
    """Get the content image features."""
    outputs = model(image)
    return outputs[NUM_STYLE_LAYERS:]


def get_style_loss(features, targets):
    """Calculates style loss."""
    return tf.reduce_mean(tf.square(features - targets))


def get_content_loss(features, targets):
    """Calculates content loss."""
    return 0.5 * tf.reduce_sum(tf.square(features - targets))


def get_style_content_loss(style_targets, style_outputs, content_targets, content_outputs, style_weight, content_weight):
    """Combine the style and content loss."""
    style_loss = tf.add_n([get_style_loss(style_output, style_target) for style_output, style_target in zip(
        style_outputs, style_targets)]) * style_weight / NUM_STYLE_LAYERS
    content_loss = tf.add_n([get_content_loss(content_output, content_target) for content_output, content_target in zip(
        content_outputs, content_targets)]) * content_weight / NUM_CONTENT_LAYERS
    return style_loss + content_loss


def calculate_gradients(image, style_targets, content_targets, style_weight, content_weight):
    """Calculate the gradients of the loss with respect to the generated image."""
    with tf.GradientTape() as tape:
        style_features = get_style_image_features(
            image, vgg_model(output_layers))
        content_features = get_content_image_features(
            image, vgg_model(output_layers))
        loss = get_style_content_loss(
            style_targets, style_features, content_targets, content_features, style_weight, content_weight)
    return tape.gradient(loss, image)


def update_image_with_style(image, style_targets, content_targets, style_weight, content_weight, optimizer):
    """Update the generated image based on the gradients."""
    gradients = calculate_gradients(
        image, style_targets, content_targets, style_weight, content_weight)
    optimizer.apply_gradients([(gradients, image)])
    image.assign(tf.clip_by_value(image, 0.0, 255.0))


def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4, epochs=1, steps_per_epoch=1):
    """Performs neural style transfer."""
    images = []
    style_targets = get_style_image_features(
        style_image, vgg_model(output_layers))
    content_targets = get_content_image_features(
        content_image, vgg_model(output_layers))

    generated_image = tf.Variable(tf.cast(content_image, dtype=tf.float32))
    images.append(content_image)

    for n in range(epochs):
        for m in range(steps_per_epoch):
            update_image_with_style(generated_image, style_targets, content_targets,
                                    style_weight, content_weight, tf.keras.optimizers.Adam())
            if (m + 1) % 10 == 0:
                images.append(generated_image)
        display_image = deprocess_image(generated_image)
        plt.imshow(display_image)
        plt.axis('off')
        plt.show()

    return tf.cast(generated_image, dtype=tf.uint8), images


def run_style_transfer(content_path, style_path, epochs=10, steps_per_epoch=100):
    """Execute the style transfer process."""
    content_image = load_and_preprocess_image(content_path)
    style_image = load_and_preprocess_image(style_path)

    generated_image, images = fit_style_transfer(
        style_image,
        content_image,
        style_weight=2e-2,
        content_weight=1e-2,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    display_image = deprocess_image(generated_image)
    plt.imshow(display_image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    run_style_transfer('./dog.png', './star.png',
                       epochs=10, steps_per_epoch=100)
