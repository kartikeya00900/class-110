# import the opencv library
import cv2
import numpy as np
import tensorflow as tf
#from keras.models import load_model

#model = load_model("keras_model.h5")

model=tf.keras.models.node_model("keras_model.h5")

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    check, frame = vid.read()

    #modify The InputPy
    #1.Resizing the image
    img=cv2.resize(frame,(224,224))

    #2.Converting The Image into numpy array and increase dimension
    test_image=np.array(img,dtype=np.float32)
    test_image=np.extends_dims(test_image,axis=0)

    #Normalsing the Image

    normalised_image = test_image/255.0



    #Prediction Result
    prediction=model.predict(normalised_image)
    print("Prediction : ",prediction)
  

    # Display the resulting frame
    cv2.imshow('result', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()