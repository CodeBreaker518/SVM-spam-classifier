from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Función para extraer características adicionales específicas para emails
def extract_features(text):
    features = {}
    
    # Características básicas
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Características de formato de email
    features['has_subject'] = 1 if 'Subject:' in text else 0
    features['subject_length'] = len(re.findall(r'Subject:.*?(?=\n)', text, re.DOTALL))
    features['has_cc'] = 1 if 'cc:' in text.lower() or 'cc :' in text.lower() else 0
    features['has_bcc'] = 1 if 'bcc:' in text.lower() or 'bcc :' in text.lower() else 0
    
    # Características de contenido
    features['special_chars'] = len(re.findall(r'[!?$%&*()]', text))
    features['numbers'] = len(re.findall(r'\d', text))
    features['uppercase_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
    features['urls'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    
    # Palabras clave específicas de emails corporativos
    business_words = ['meeting', 'report', 'project', 'deadline', 'team', 'client', 'business', 'market', 'sales']
    features['business_words'] = sum(1 for word in business_words if word in text.lower())
    
    # Palabras clave de urgencia
    urgency_words = ['urgent', 'immediately', 'asap', 'deadline', 'important', 'priority', 'critical']
    features['urgency_words'] = sum(1 for word in urgency_words if word in text.lower())
    
    # Palabras clave de spam en emails corporativos
    spam_words = ['click here', 'unsubscribe', 'limited time', 'exclusive offer', 'act now', 'congratulations', 'winner']
    features['spam_words'] = sum(1 for word in spam_words if word in text.lower())
    
    # Características de estructura
    features['paragraphs'] = len(text.split('\n\n'))
    features['sentences'] = len(re.split(r'[.!?]+', text))
    features['avg_sentence_length'] = features['word_count'] / max(features['sentences'], 1)
    
    return features

# Cargar datos del archivo CSV
print("\n" + "="*50)
print("FASE 1: CARGA DE DATOS")
print("="*50)
print("\nCargando dataset de emails spam...")
df = pd.read_csv('spam.csv')
print("\nDistribución de clases en el dataset:")
print("Ham (No Spam): {:.2%}".format(df['label'].value_counts(normalize=True)['ham']))
print("Spam: {:.2%}".format(df['label'].value_counts(normalize=True)['spam']))

# Preparar datos
emails = df['text'].values
labels = df['label_num'].values

# Crear pipeline con vectorizador y clasificador
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),
        max_features=3000,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r'(?u)\b\w\w+\b|#\d+'  # Mejorar detección de números de orden
    )),
    ('classifier', LinearSVC(
        C=0.3,              # Más conservador
        class_weight={0: 1.3, 1: 0.7},  # Más peso a NO SPAM
        max_iter=10000,
        dual=False
    ))
])

# Dividir datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.3, random_state=42, stratify=labels)

print("\n" + "="*50)
print("FASE 2: ENTRENAMIENTO DEL MODELO")
print("="*50)
print("\nEntrenando modelo SVM...")

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Evaluar modelo
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión: {accuracy:.2f}")
print("\nReporte detallado:")
print(classification_report(y_test, y_pred, target_names=['No Spam', 'Spam']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Spam', 'Spam'], yticklabels=['No Spam', 'Spam'])
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.savefig('confusion_matrix.png')
plt.close()

# Función para clasificar nuevos emails
def classify_email(email):
    prediction = pipeline.predict([email])[0]
    decision_function = pipeline.decision_function([email])[0]
    probability = 1 / (1 + np.exp(-decision_function))  # Convertir a probabilidad
    
    result = "SPAM" if prediction == 1 else "NO SPAM"
    confidence = probability if prediction == 1 else 1 - probability
    
    return f"'{email}' -> {result} (confianza: {confidence:.2f})"

# Ejemplos de uso
print("\n" + "="*50)
print("FASE 4: CLASIFICACIÓN DE NUEVOS EMAILS")
print("="*50)
print("\nAnalizando los siguientes emails:")
print("-"*30)

test_emails = [
    """Subject: RE: Follow-up on Yesterday's Discussion
Hi Team,
Just wanted to follow up on our discussion about the quarterly budget review. 
Could you please share the updated financial projections by EOD?
Best regards,
Finance Director""",

    """Subject: Your Package Delivery Status
Dear Customer,
Your package #DEL123456 has been delayed due to weather conditions.
Expected delivery date: Tomorrow
Track your package here: http://tracking.example.com
Customer Service""",

    """Subject: URGENT: Your Account Will Be Suspended
Dear User,
We detected suspicious activity in your account. Your account will be suspended in 24 hours unless you verify your information now.
Click here to verify: http://verify.example.com
Security Team""",

    """Subject: Monthly Team Performance Review
Hi All,
Please find attached the monthly performance metrics for your review.
Key highlights:
- Sales targets achieved: 95%
- Customer satisfaction: 4.8/5
- Project milestones: On track
Let's discuss in our next meeting.
Regards,
Team Lead""",

    """Subject: You've Been Selected for Our Elite Program!
Congratulations! You're among the 0.1% selected for our exclusive VIP program!
Get instant access to premium features and special discounts!
Limited time offer - Click here to claim your benefits now!
VIP Department"""
]

# Función para mostrar los emails disponibles
def mostrar_emails_disponibles():
    print("\nEmails disponibles para análisis:")
    print("-"*30)
    for i, email in enumerate(test_emails, 1):
        subject = email.split('\n')[0].replace('Subject: ', '')
        print(f"{i}. {subject}")
    print("-"*30)

# Función para analizar un email específico
def analizar_email_seleccionado(numero_email):
    if 1 <= numero_email <= len(test_emails):
        email = test_emails[numero_email - 1]
        print("\n" + "="*50)
        print(f"ANALIZANDO EMAIL #{numero_email}")
        print("="*50)
        print("\nContenido del email:")
        print("-"*30)
        print(email)
        print("-"*30)
        print("\n¿Es este email SPAM o NO SPAM?")
        print("(Presiona Enter para ver la clasificación del modelo)")
        input()
        result = classify_email(email)
        print(f"\nClasificación del modelo: {result}")
    else:
        print("\nNúmero de email inválido. Por favor selecciona un número entre 1 y", len(test_emails))

# Menú principal
def menu_principal():
    while True:
        print("\n" + "="*50)
        print("MENÚ PRINCIPAL")
        print("="*50)
        print("\n1. Ver emails disponibles")
        print("2. Analizar un email específico")
        print("3. Salir")
        print("\nSelecciona una opción (1-3): ", end='')
        
        opcion = input().strip()
        
        if opcion == '1':
            mostrar_emails_disponibles()
        elif opcion == '2':
            mostrar_emails_disponibles()
            print("\nIngresa el número del email que quieres analizar: ", end='')
            try:
                numero = int(input().strip())
                analizar_email_seleccionado(numero)
            except ValueError:
                print("\nPor favor ingresa un número válido.")
        elif opcion == '3':
            break
        else:
            print("\nOpción inválida. Por favor selecciona 1, 2 o 3.")

# Iniciar el programa
if __name__ == "__main__":
    print("\n" + "="*50)
    print("DINÁMICA: CLASIFICACIÓN DE SPAM")
    print("="*50)
    
    menu_principal()