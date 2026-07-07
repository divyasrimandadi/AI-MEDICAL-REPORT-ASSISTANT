import random

print("DEMO PREDICTOR LOADED")


def predict_image(image):

    pneumonia_confidence = round(
        random.uniform(75, 99),
        2
    )

    normal_confidence = round(
        100 - pneumonia_confidence,
        2
    )

    if pneumonia_confidence > 85:

        prediction = "PNEUMONIA"

        confidence = pneumonia_confidence

    else:

        prediction = "NORMAL"

        confidence = normal_confidence

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        confidence
    )

    return prediction, confidence