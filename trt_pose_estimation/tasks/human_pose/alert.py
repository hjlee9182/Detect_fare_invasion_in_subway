# -*- coding: utf-8 -*-
import requests

def notifier():
    TARGET_URL = 'https://notify-api.line.me/api/notify'
    TOKEN = '0V1CTaDiJKYBGRvynfxEvRielmXM7juxFKwD4SVXeuk'
    image = open('output/crop.jpg','rb')

    response = requests.post(
      TARGET_URL,
      headers={
        'Authorization': 'Bearer ' + TOKEN
      },
      data={
        'message': '[*] 이상행동이 감지되었습니다!!!',
      },
      files={
          'imageFile': image
      }
    )
    return response.text


