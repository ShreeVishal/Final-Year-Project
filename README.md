# Final-Year-Project
This is my implementation of an image segmentation model (uNet + segNet) that automates object region delineation based on user input image into the web application developed with the Flask micro web framework.

Files within the repository, the purposes they serve and the implementation techniques wherever necessary:
1. UNET.ipynb -> Google Colab Notebook used to train the uNet model, you may load the file to observe the training techniques implemented.
2. SegNet.ipynb -> Google Colab Notebook used to train the segNet model, you may load the file to observe the training techniques implemented.
3. .vscode, __pycache__, static, templates, app.py -> Neccessary files to load into Visual Studio Code and execute the implementations of the web application.
4. Final Year Project Report.pdf -> Content report on my Final Year Project (contains in-depth explanation on model training, model deployment and other features alongside justifications for techniques used)

Note (app.py):
1. Line 98, __unet_model.load_weights("C:\\Users\\Vishal\\Flask Project\\Unet_Testing.h5")__ and line 125, __segnet_model.load_weights("C:\\Users\\Vishal\\Flask Project\\Segnet_Testing.h5")__ are responsible for loading the trained image segmentation model into the web application. These files can be generated from Google Colab via the __ModelCheckpoint()__ function which saves the trained model into a __.h5__ file that can then be used in any other external environment. 

__Implementation screenshots:__
![image](https://github.com/ShreeVishal/Final-Year-Project/assets/93562563/29836603-e0c4-4356-a134-be61bdd28e78) 
- Web Application Homepage

  
![image](https://github.com/ShreeVishal/Final-Year-Project/assets/93562563/62e2edeb-0b42-4e2f-b0b9-045541241c1e) 
- Prompt user input for an image of their choice

  
![image](https://github.com/ShreeVishal/Final-Year-Project/assets/93562563/417a4565-a9af-4c5b-9d52-48d279288d02) 
- Display input image for user verification

  
![image](https://github.com/ShreeVishal/Final-Year-Project/assets/93562563/1d034c5d-54a4-4a99-9b84-412e39681f02)
![image](https://github.com/ShreeVishal/Final-Year-Project/assets/93562563/2f888ee7-ac68-43ef-90a5-32ac5a76cb03) 
- Predicted image for uNet and segNet model respectively

__How to load the app?__:
Once downloading the files (.vscode, pycache, static, templates, app.py), ensure that they are located within the same directory and execute the app.py script like so ![image](https://github.com/ShreeVishal/Final-Year-Project/assets/93562563/226dbad4-7c49-42c7-9b1e-9997b1386009)

Flask should then display a localhost IP address that can be used within any web browser to display the implementation of the web application.
![image](https://github.com/ShreeVishal/Final-Year-Project/assets/93562563/3e30d2c1-9cea-4e63-9a07-4bc6c610580b)



