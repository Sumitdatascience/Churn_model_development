import unittest
from flask_app.app import app
import json

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        """Test if the home page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Churn Prediction Form</title>', response.data)

    def test_predict_page(self):
        """Test the /predict endpoint with valid input data"""
        # Example input data in JSON format matching the expected model features
        input_data = {
            'tenure': 12,
            'preferredLoginDevice': 'desktop',
            'cityTier': 1,
            'warehouseToHome': 2,
            'preferredPaymentMode': 'credit card',
            'gender': 'male',
            'hourSpendOnApp': 2,
            'numberOfDeviceRegistered': 3,
            'preferredOrderCat': 'fashion',
            'satisfactionScore': 2,
            'maritalStatus': 0,
            'numberOfAddress': 3.0,
            'complain': 0,
            'orderAmountHikeFromLastYear': 13.0,
            'orderCount': 10,
            'daySinceLastOrder': 20.0,
            'cashbackAmount': 160.0
        }

        # Send POST request to /predict with the input data
        response = self.client.post('/predict', 
                                    data=json.dumps(input_data),
                                    content_type='application/json')

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the prediction is either '0' or '1'
        response_data = json.loads(response.data)
        self.assertIn(response_data['prediction'], ['0', '1'], "Prediction should be either '0' or '1'")

if __name__ == '__main__':
    unittest.main()
