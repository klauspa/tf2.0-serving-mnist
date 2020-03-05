# tf2.0-serving-mnist
use tensorflow2.0 serving to serve simple mnist model --only 100 lines of code

# dependency
flask

tensorflow==2.0.0

tensorflow-model-server

# how to use
1.python train.py
  train the model and export pb file
  
2.tensorflow_model_server --rest_api_port=8501 --model_name=saved_model --model_base_path=/.../saved_model
  depoly the model to tensorflow serving
  
3.python app.py 
  run flask 
  then you are ready to go
