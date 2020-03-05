from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import re
import base64
import sys 
import os
import requests
import json

#initalize our flask app
app = Flask(__name__)

#tensorflow serving URL
URL = "http://localhost:8501/v1/models/saved_model:predict"

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(b',(.*)',imgData1).group(1) # 匹配第一个括号

	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))
	
@app.route('/')
def index():
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#get the raw data format of the image
	imgData = request.get_data() 

	#encode it into a suitable format
	convertImage(imgData)

	#read the image into memory
	x = imread('output.png',mode='L')

	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))

	#convert to a 3D tensor to feed into our model
	x = x.reshape(1,28,28)
	 
	#call tensorflow serving
	tf_serving_request = json.dumps({"signature_name": "serving_default",
                                     "instances": x.tolist()})
	
	resp = requests.post(URL, data=tf_serving_request)

	#200 tensorflow serving成功返回结果
	print('response.status_code: {}'.format(resp.status_code))

	predictions = np.array(json.loads(resp.text)["predictions"])

	response = np.array_str(np.argmax(predictions,axis=1))
	return response	

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	app.run(debug=True)
