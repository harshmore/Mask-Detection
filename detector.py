import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

detector = load_model('detection.h5')
model = load_model('mask.h5')
UPLOAD_FOLDER = 'static'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def model_predict(imgpath):
    path = os.path.join('images',imgpath)
    img = cv2.imread(path)
    
    img = cv2.resize(img,(224,224))
    orig_img = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    new_img = img.reshape((1,224,224,3))
    new_img = new_img/255
    predictions = detector.predict(new_img)
    predictions = predictions.reshape((1,5,4))

    for notations in predictions[0]:
        [xmin,ymin,xmax,ymax] = notations
        new_img = img[int(ymin*224):int(ymax*224),int(xmin*224):int(ymax*224)]
        
        try:
            new_img = cv2.resize(new_img,(128,128))
            new_img = new_img/255
        except:
            continue
        pred = model.predict(new_img.reshape(-1,128,128,3))
        print(pred)
        cv2.rectangle(orig_img,(int(xmin*224)-10,int(ymin*224)-10),(int(xmax*224)+10,int(ymax*224)+10),(0,255,0),2)
        if pred[0][0]>0.90:
            text='Masked'
        else:
            text = 'Not Masked'
        cv2.putText(orig_img,text,org=(int(xmin*224),int(ymax*224)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color = (0,0,255))
            


    
    path = os.path.join('static',imgpath)
    cv2.imwrite(path,orig_img)
    
    return path


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        path = model_predict(filename)[8:]
        
        print(path)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], path))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
#print('display_image filename: ' + filename)
    return redirect(url_for('static', filename= filename), code=301)

if __name__ == "__main__":
    app.run()