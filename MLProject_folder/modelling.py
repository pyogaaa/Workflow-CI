import pandas as pd
import numpy as np
import os
import mlflow
import dagshub
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    if os.getenv('GITHUB_ACTIONS'):
        # Membaca dari environment variable di file YAML
        remote_url = os.getenv('MLFLOW_TRACKING_URI')
        mlflow.set_tracking_uri(remote_url)
    else:
        # Tetap bisa jalan normal kalau kamu running lokal
        dagshub.init(repo_owner='pyogaaa', repo_name='SMSML_Yoga-pratama', mlflow=True)
    
    # 2. AKTIFKAN AUTOLOG
    # Otomatis mencatat parameter, metrik, dan model scikit-learn
    mlflow.sklearn.autolog()

    # Path dinamis untuk file dan folder output
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'output', 'modelling')
    os.makedirs(output_dir, exist_ok=True) # Memastikan folder output/modelling ada

    data_path = os.path.join(base_dir, 'indonesian_crime_tweets_preprocessed.csv')
    
    if not os.path.exists(data_path):
        print(f"[!] File {data_path} tidak ditemukan!")
        return

    # Load Data
    df = pd.read_csv(data_path)
    df['processed_text'] = df['processed_text'].fillna('missing')

    # Feature Engineering
    X_raw = df[['processed_text', 'user_followers', 'user_friends', 'retweet_count', 'favorite_count']]
    y = df['label'].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vektorisasi Teks
    tfidf = TfidfVectorizer(max_features=2000)
    X_train_tfidf = tfidf.fit_transform(X_train_raw['processed_text']).toarray()
    X_test_tfidf = tfidf.transform(X_test_raw['processed_text']).toarray()

    # Menggabungkan fitur teks (TF-IDF) dengan fitur numerik lainnya
    X_train = np.hstack((X_train_tfidf, X_train_raw[['user_followers', 'user_friends', 'retweet_count', 'favorite_count']].values))
    X_test = np.hstack((X_test_tfidf, X_test_raw[['user_followers', 'user_friends', 'retweet_count', 'favorite_count']].values))

    # 3. Training Model
    with mlflow.start_run(run_name="Base_Model_RF_Autolog", nested=True):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # --- PENYIMPANAN FISIK DI SUB-FOLDER OUTPUT ---
        model_save_path = os.path.join(output_dir, 'model_base.pkl')
        tfidf_save_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
        
        joblib.dump(model, model_save_path)
        joblib.dump(tfidf, tfidf_save_path)

        # Log file tersebut juga sebagai artefak di MLflow
        mlflow.log_artifact(model_save_path)
        mlflow.log_artifact(tfidf_save_path)

        print(f"\n--- BASE MODEL LOGGED TO LOCAL & DAGSHUB ---")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"File output tersimpan di: {output_dir}")
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="Crime_Model" # Nama ini harus sama dengan yang di YAML
        )

if __name__ == "__main__":
    main()