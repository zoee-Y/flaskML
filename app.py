from flask import Flask, request
from EmotionPredictor import EmotionPredictorByFacialLandmarks
from utils import utils

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    npyFLP = utils.convertToNumpyArr(request.form.values())
    emotionPredictor = EmotionPredictorByFacialLandmarks('./fpModel.pkl')
    prediction = emotionPredictor.predictEmotion(npyFLP)
    predictionList = prediction.tolist()
    t = ' ,'.join(str(n) for n in predictionList)
    print('prediction == ', t)

    return t


if __name__ == '__main__':
    app.run()
