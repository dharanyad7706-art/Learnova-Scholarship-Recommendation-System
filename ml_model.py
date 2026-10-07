import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Sample training dataset
data = {
    "percentage": [
        55, 60, 65, 70, 75,
        80, 85, 90, 92, 95
    ],

    "income": [
        600000, 500000, 450000, 400000, 350000,
        300000, 250000, 200000, 150000, 100000
    ],

    "eligible": [
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}


df = pd.DataFrame(data)


# Features
X = df[
    [
        "percentage",
        "income"
    ]
]


# Target
y = df["eligible"]


# Create ML model
model = DecisionTreeClassifier(
    random_state=42
)


# Train model
model.fit(X, y)


def predict_eligibility(
    percentage,
    income
):

    input_data = pd.DataFrame(
        [
            {
                "percentage": percentage,
                "income": income
            }
        ]
    )

    prediction = model.predict(
        input_data
    )

    return int(
        prediction[0]
    )


# Test
if __name__ == "__main__":

    result = predict_eligibility(
        82,
        250000
    )

    print("ML Prediction:", result)