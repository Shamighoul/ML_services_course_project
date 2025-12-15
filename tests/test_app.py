import pytest
import requests

from tests.test_ml import load_test_images


@pytest.mark.parametrize("image, expected_label", load_test_images())
def test_digit(input_image: str, expected_label: str):
    response = requests.get("http://0.0.0.0/forward/", params={"image": input_image})
    assert response.json()["image_base64"] == input_image
    assert response.json()["label"] == expected_label
