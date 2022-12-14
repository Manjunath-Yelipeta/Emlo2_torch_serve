
import unittest

import requests
import json
import base64
from requests import Response
import os


class TestRestApiInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://44.234.35.197:8080/predictions/cifar/1.0"


        cls.image_paths = os.listdir("/home/ubuntu/Emlo2_torch_serve/test_serve/cifar_images/")

        # convert image to base64

    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")

            data = open('/home/ubuntu/Emlo2_torch_serve/test_serve/cifar_images/' + image_path, 'rb').read()
              
            response: Response = requests.request("POST", self.base_url, data=data, timeout=15)

            print(f"response: {response.text}")

            data = response.json()

            predicted_label = list(data)[0]
            act_label = image_path.split(".")[0].split('_')[-1]

            print(f"predicted label: {predicted_label}, actual label: {act_label}")

            self.assertEqual(act_label, predicted_label)

            print(f"done testing: {image_path}")

            print()


if __name__ == '__main__':
    unittest.main()
