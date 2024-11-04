# Neural Style Transfer Web App

This project is a Streamlit-based web application that applies neural style transfer to combine the style of one image with the content of another. Users can upload their own images or select from default styles, and the app will generate a styled output image. The style transfer is based on a VGG19 network, delivering unique and visually artistic outputs.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Disclaimer](#disclaimer)

## Features
- **Upload Content and Style Images**: Allows users to upload custom content and style images.
- **Default Style Options**: Choose from iconic styles like "Starry Night" and "The Scream."
- **Customizable Parameters**: Users can adjust the number of epochs, steps per epoch etc. to control the stylization detail.
- **Image Download**: Styled images can be downloaded directly from the app.

## Installation
1. **Clone the Repository**
    ```bash
    git clone https://github.com/tapan2003/Art-Style-Transfer.git
    cd Art-Style-Transfer

2. **Install Dependencies** 
   Ensure you have Python and pip installed, then run:
    ```bash
    pip install -r requirements.txt

3. **Prepare Style Images Folder** 
    Make sure you have a styles folder in the project root, containing default style images as specified in app.py.

## Usage
To run the app, use the following command:
    ```bash
    streamlit run app.py
    
This will open the Streamlit interface in your default web browser.

## Disclaimer
This application performs best when run on a machine with a GPU. Running the Neural Style Transfer process on a CPU may result in significantly longer processing times, especially with high values for epochs and steps per epoch.