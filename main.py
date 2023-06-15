import uvicorn
from fastapi import FastAPI, UploadFile, File, jsonify
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import shutil
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI()  # create a new FastAPI app instance

port = os.environ.get("PORT", 8080)
#port = 8080

model = tf.keras.models.load_model('model2_wisnu.h5')

def predict(file):
    img = image.load_img(file, target_size=(150,150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x]) 
    classes = model.predict(images) 
    print(classes[0]) 

    result = np.argmax(classes[0])
    label = ['freshapples', 'freshbanana', 'freshoranges','rottenapples','rottenbanana','rottenoranges']
    
    return label[result]

@app.get("/")
def hello_world():
    return ("hello world")
    
@app.post("/predict")
def classify(input: UploadFile = File(...)):
    print(input.filename)
    print(type(input.filename))
    savefile = input.filename
    with open(savefile, "wb") as buffer:
        shutil.copyfileobj(input.file, buffer)
    result = predict(savefile)
    os.remove(savefile)

    response = { "result": result }

    return jsonify(response)
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
