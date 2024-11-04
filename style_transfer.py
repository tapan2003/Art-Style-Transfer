import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_and_preprocess_image(image_path, max_dim=512):
    img = load_img(image_path)
    img = img_to_array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = np.expand_dims(img, axis=0).copy()
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_image(image):
    image = image.numpy().squeeze()
    image = image[:, :, ::-1]  # Convert from BGR to RGB
    image += [103.939, 116.779, 123.68]
    return np.clip(image, 0, 255).astype('uint8')


def save_image(image, path):
    img = deprocess_image(image.numpy())
    img = Image.fromarray(img)
    img.save(path)


style_layers = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']
output_layers = style_layers + content_layers
NUM_CONTENT_LAYERS = len(content_layers)
NUM_STYLE_LAYERS = len(style_layers)


def vgg_model(layer_names):
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)


def gram_matrix(input_tensor):
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.shape(input_tensor)[
                            1] * tf.shape(input_tensor)[2], tf.float32)
    return gram / num_locations


def get_style_image_features(image, model):
    outputs = model(image)
    return [gram_matrix(style_layer) for style_layer in outputs[:NUM_STYLE_LAYERS]]


def get_content_image_features(image, model):
    outputs = model(image)
    return outputs[NUM_STYLE_LAYERS:]


def calculate_gradients(image, style_targets, content_targets, style_weight, content_weight):
    with tf.GradientTape() as tape:
        style_features = get_style_image_features(
            image, vgg_model(output_layers))
        content_features = get_content_image_features(
            image, vgg_model(output_layers))
        style_loss = tf.add_n([tf.reduce_mean(tf.square(style_output - style_target)) for style_output,
                              style_target in zip(style_features, style_targets)]) * style_weight / NUM_STYLE_LAYERS
        content_loss = tf.add_n([tf.reduce_sum(tf.square(content_output - content_target)) for content_output,
                                content_target in zip(content_features, content_targets)]) * content_weight / NUM_CONTENT_LAYERS
        loss = style_loss + content_loss
    return tape.gradient(loss, image)


def update_image_with_style(image, style_targets, content_targets, style_weight, content_weight, optimizer):
    gradients = calculate_gradients(
        image, style_targets, content_targets, style_weight, content_weight)
    optimizer.apply_gradients([(gradients, image)])
    image.assign(tf.clip_by_value(image, 0.0, 255.0))


def run_style_transfer(content_path, style_path, epochs=10, steps_per_epoch=100, style_weight=1e-2, content_weight=1e-4, progress_bar=None):
    content_image = load_and_preprocess_image(content_path)
    style_image = load_and_preprocess_image(style_path)
    style_targets = get_style_image_features(
        style_image, vgg_model(output_layers))
    content_targets = get_content_image_features(
        content_image, vgg_model(output_layers))
    generated_image = tf.Variable(tf.cast(content_image, dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam()

    for n in range(epochs):
        for m in range(steps_per_epoch):
            update_image_with_style(generated_image, style_targets,
                                    content_targets, style_weight, content_weight, optimizer)
            if progress_bar:
                progress_bar.progress(
                    (n * steps_per_epoch + m + 1) / (epochs * steps_per_epoch))

    return generated_image
