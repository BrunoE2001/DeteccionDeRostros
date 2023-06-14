# Inicio
import os.path
import numpy as np
import cv2
import json
from flask import Flask, request, Response
import uuid

#Funcion Detectar Rostro de una imagen
# Usar interpolacion bicubica
def faceDetect(img):
    face_cascade = cv2.CascadeClassifier('face_detect_cascade.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,255))
        # Guardar Archivo
        path_file = ('static/%s.jpg' %uuid.uuid4().hex)
        cv2.imwrite(path_file, img)
        # Devuelve el nombre de archivo de imagen
        return json.dumps(path_file)

# API
ByM = Flask(__name__)

# Enrutar HTTP POST a este método
@ByM.route('/api/upload', methods=['POST'])
def upload():
    # Recibimos la imagen del cliente
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # Procesar imagen
    img_processed = faceDetect(img)
    # Responder
    return Response(response=img_processed, status=200, mimetype="application/json") # Devolver en Json

# Iniciar Servidor
ByM.run(host="0.0.0.0", port=5000)