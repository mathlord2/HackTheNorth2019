import requests
import json

def getImg(image_url):
    # set to your own subscription key value
    subscription_key = 'fb8d8fc66323471fb444337a362856c6'
    assert subscription_key

    # replace <My Endpoint String> with the string from your endpoint URL
    face_api_url = 'https://emoticapture.cognitiveservices.azure.com/face/v1.0/detect'

    headers = {'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': subscription_key}

    data = open(image_url, 'rb')

    params = {
        'returnFaceId': 'false',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'emotion'
    }

    response = requests.post(face_api_url, params=params, headers=headers, data=data)

    return response.json()