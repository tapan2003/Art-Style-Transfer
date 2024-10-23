import streamlit as st
from PIL import Image
import os
from style_transfer import run_style_transfer, save_image

# Set the default style images (make sure these images are in the same directory or accessible)
DEFAULT_STYLES = {
    "Starry Night": "./styles/starry_night.jpg",
    "The Scream": "./styles/the_scream.jpg",
    "Great Wave of Kanagawa": "./styles/great_wave_of_kanagawa.jpg"
}

# App title
st.title("Neural Style Transfer Web App")

# Sidebar options for user input
st.sidebar.header("Upload or Select Style Image")
content_file = st.file_uploader(
    "Upload a Content Image", type=["jpg", "jpeg", "png"])
style_option = st.sidebar.selectbox("Select a Default Style or Upload Your Own",
                                    ["Select a Default Style"] + list(DEFAULT_STYLES.keys()) + ["Upload Custom Style Image"])

# Handle style image upload or selection
if style_option == "Upload Custom Style Image":
    style_file = st.file_uploader(
        "Upload a Style Image", type=["jpg", "jpeg", "png"])
else:
    style_file = None

# Parameters for neural style transfer
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=10)
steps_per_epoch = st.sidebar.slider(
    "Steps per Epoch", min_value=50, max_value=500, value=100)

# Process the style transfer once images are provided
if content_file is not None:
    content_image = Image.open(content_file)

    # Get the style image either from the selection or user upload
    if style_option in DEFAULT_STYLES:
        style_image_path = DEFAULT_STYLES[style_option]
        style_image = Image.open(style_image_path)
    elif style_file is not None:
        style_image = Image.open(style_file)
    else:
        style_image = None

    if style_image is not None:
        st.write("### Content Image")
        st.image(content_image, use_column_width=True)

        st.write("### Style Image")
        st.image(style_image, use_column_width=True)

        # Button to start style transfer
        if st.button("Run Style Transfer"):
            with st.spinner("Generating styled image..."):
                # Save the uploaded content and style images temporarily
                content_image_path = "./temp_content_image.jpg"
                style_image_path = "./temp_style_image.jpg"
                content_image.save(content_image_path)
                style_image.save(style_image_path)

                # Run the style transfer (using the imported function)
                output_image = run_style_transfer(
                    content_image_path, style_image_path, epochs, steps_per_epoch)

                # Save the output image
                output_image_path = "./styled_output.jpg"
                save_image(output_image, output_image_path)

                # Display the result
                st.write("### Styled Image")
                st.image(output_image, use_column_width=True)

                # Option to download the styled image
                with open(output_image_path, "rb") as file:
                    st.download_button(label="Download Styled Image", data=file,
                                       file_name="styled_image.jpg", mime="image/jpeg")
    else:
        st.warning("Please select or upload a style image.")

else:
    st.info("Please upload a content image to begin.")
