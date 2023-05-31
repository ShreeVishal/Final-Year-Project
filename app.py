from flask import Flask, render_template, request, session
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
from werkzeug.utils import secure_filename
from waitress import serve
#*** Backend operation
 
# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templates', static_folder='static')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 
#Defining the Unet Model
def double_conv_block(x, n_filters):

   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)

   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)

   return x

 # inputs
inputs = layers.Input(shape=(128,128,3))

   # encoder: contracting path - downsample
   # 1 - downsample
f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
f4, p4 = downsample_block(p3, 512)

   # 5 - bottleneck
bottleneck = double_conv_block(p4, 1024)

   # decoder: expanding path - upsample
   # 6 - upsample
u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
u9 = upsample_block(u8, f1, 64)

   # outputs
outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

   # unet model with Keras Functional API
unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

#Loading the trained weights from Google Colab into the model defined in Flask (Unet)
unet_model.load_weights("C:\\Users\\Vishal\\Flask Project\\Unet_Testing.h5")

#Defining the Segnet Model
# Encoder
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)

# Decoder
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
up6 = UpSampling2D(size=(2, 2))(conv5)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
up7 = UpSampling2D(size=(2, 2))(conv6)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
up8 = UpSampling2D(size=(2, 2))(conv7)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)

outputs = layers.Conv2D(3, 1, padding="same", activation='softmax')(conv8)

segnet_model = Model(inputs=inputs, outputs=outputs)
segnet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Loading the trained weights from Google Colab into the model defined in Flask (Segnet)
segnet_model.load_weights("C:\\Users\\Vishal\\Flask Project\\Segnet_Testing.h5")

def preprocess_image(image):
    image = Image.open(image)
    image = image.convert('RGB')
    image = image.resize((128, 128))
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = image[np.newaxis,:, :, :]
    return image

def create_mask(pred_mask):
 pred_mask = tf.argmax(pred_mask, axis=-1)
 pred_mask = pred_mask[..., tf.newaxis]
 return pred_mask[0]

@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('index_upload_and_display_image_page2.html')
 
@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', None)
    # Display image in Flask application web page
    return render_template('show_image.html', user_image = img_file_path)

#function to return Unet Predicted Image
@app.route('/predicted_image')
def showPredictedImage():
    image = session.get('uploaded_img_file_path', None)
    image1 = preprocess_image(image)
    image2 = unet_model.predict(image1)
    image3 = create_mask(image2)
    #plt.imshow(image3)
    predicted_image = plt.imshow(image3)
    image_filename = "predicted_image.jpeg"
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],image_filename))

    return render_template('predicted_image.html')

#function to return Segnet Predicted Image
@app.route('/segnet_image')
def showSegnetPrediction():
   image = session.get('uploaded_img_file_path', None)
   image1 = preprocess_image(image)
   image2 = segnet_model.predict(image1)
   image3 = create_mask(image2)
   #plt.imshow(image3)
   segnet_image = plt.imshow(image3)
   image_filename = "segnet_image.jpeg"
   plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))

   return render_template('segnet_image.html')


#Server hosting: port number 5000, 5500 with Live Server
if __name__=='__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app)
   


