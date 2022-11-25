import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)
sys.path.insert(0,'serve/ts_scripts/')
import unittest
import os

import requests
import json
import base64
from requests import Response
from serve.ts_scripts.torchserve_grpc_client import infer, get_inference_stub

class TestRestApiInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):


        cls.image_paths = os.listdir("/home/ubuntu/Emlo2_torch_serve/test_serve/cifar_images/")
        
        cls.stub = get_inference_stub()
        # convert image to base64
    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")

            response = infer(self.stub, 'cifar', '/home/ubuntu/Emlo2_torch_serve/test_serve/cifar_images/' + image_path)
   

            # print(f"response: {response.text}")
            data = json.loads(response)

            predicted_label = list(data)[0]
            act_label = image_path.split(".")[0].split('_')[-1]

            print(f"predicted label: {predicted_label}, actual label: {act_label}")

            self.assertEqual(act_label, predicted_label)

            print(f"done testing: {image_path}")

            print()


if __name__ == '__main__':
    unittest.main()