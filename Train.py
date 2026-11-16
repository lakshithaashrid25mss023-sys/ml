import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("C:\ml\Dataset\student.csv")


df = df.dropna()

X = df[['study_hours', 'attendance', 'previous_marks']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")


new_student = pd.DataFrame({
    'study_hours': [7],
    'attendance': [95],
    'previous_marks': [60]
})

result = model.predict(new_student)

print("Student Result:", result[0])