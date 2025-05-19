from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Cargar datos del archivo CSV
df = pd.read_csv('spam.csv', encoding='latin-1')
# Los datos tienen columnas v1 (etiquetas) y v2 (mensajes)
emails = df['v2'].values
# Convertir etiquetas a números: 'spam' = 1, 'ham' = 0
labels = (df['v1'] == 'spam').astype(int).values

# 1. Vectorización del texto usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(emails)

# 2. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 3. Crear y entrenar el modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# 4. Hacer predicciones
y_pred = svm_model.predict(X_test)

# 5. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy:.2f}")
print("\nReporte detallado:")
print(classification_report(y_test, y_pred, target_names=['No Spam', 'Spam']))

# 6. Función para clasificar nuevos emails
def classify_email(email):
    email_vectorized = vectorizer.transform([email])
    prediction = svm_model.predict(email_vectorized)[0]
    probability = svm_model.decision_function(email_vectorized)[0]
    
    result = "SPAM" if prediction == 1 else "NO SPAM"
    confidence = abs(probability)
    
    return f"'{email}' -> {result} (confianza: {confidence:.2f})"

# Ejemplos de uso
print("\n" + "="*50)
print("CLASIFICACIÓN DE NUEVOS EMAILS:")
print("="*50)

test_emails = [
    "GENT! We are trying to contact you. Last weekends draw shows that you won a £1000 prize GUARANTEED. Call 09064012160. Claim Code K52. Valid 12hrs only. 150ppm",
    "Hello, my love. What are you doing? Did you get to that interview today? Are you happy? Are you being a good boy? Do you think of me?",
    "BangBabes Ur order is on the way. U SHOULD receive a Service Msg 2 download UR content. If U do not, GoTo wap. bangb. tv on UR mobile internet/service menu",
    "Hi :)finally i completed the course:)",
    "Please call our customer service representative on FREEPHONE 0808 145 4742 between 9am-11pm as you have WON a guaranteed £1000 cash or £5000 prize!"
]

for email in test_emails:
    print(classify_email(email))