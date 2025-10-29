from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import os
import warnings
import pickle
import pandas as pd
import traceback 

warnings.filterwarnings('ignore')
app = Flask(__name__)
app.secret_key = os.urandom(24) 

USERS_FILE = "users.json"
MODELS = {} 

# --- Model Loading Logic ---
try:
    MODELS['diabetes_model'] = pickle.load(open("rf.pkl", "rb"))
    MODELS['diabetes_scaler'] = pickle.load(open("scaler.pkl", "rb"))
    MODELS['diabetes_encoder'] = pickle.load(open("encoder.pkl", "rb"))
    print("Diabetes (RF) Model, scaler, and encoder loaded successfully!")
except Exception as e:
    print(f"Error loading Diabetes assets: {e}")

try:
    MODELS['hd_model'] = pickle.load(open("hd_model.pkl", "rb"))
    print("Heart Disease Model loaded successfully!")
except FileNotFoundError:
    print("Warning: hd_model.pkl not found. Heart Disease Prediction will be unavailable.")
    MODELS['hd_model'] = None
except Exception as e:
    print(f"Error loading Heart Disease assets: {e}")
    MODELS['hd_model'] = None
    
# Check if necessary models are loaded
if not all(key in MODELS for key in ['diabetes_model', 'diabetes_scaler', 'diabetes_encoder']):
    print("FATAL: Required Diabetes assets not fully loaded.")
if MODELS['hd_model'] is None:
    print("INFO: Heart Disease prediction functions will use an uninitialized model.")



@app.route("/diabetesPrediction", methods=["POST"])
def diabetesPrediction():
    model = MODELS.get('diabetes_model')
    scaler = MODELS.get('diabetes_scaler')
    encoder = MODELS.get('diabetes_encoder')
    if model is None or scaler is None:
        return "Model or scaler not available"
    try:
        age = float(request.form.get("age"))
        gender_input = request.form.get("gender")  
        hypertension = int(request.form.get("hypertension"))
        heart_disease = int(request.form.get("heartdisease"))
        smoking_history = int(request.form.get("smokinghistory"))
        bmi = float(request.form.get("bmi"))
        HbA1c_level = float(request.form.get("HbA1c_level"))
        blood_glucose_level = float(request.form.get("bloodglucoselevel"))
        gender = 1 if gender_input == "Female" else 0
        raw_data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [['No Info', 'current', 'ever', 'former', 'never', 'not current'][smoking_history]],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level]
        }

        df_raw = pd.DataFrame(raw_data)

        numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        df_scaled = df_raw.copy()
        df_scaled[numerical_features] = scaler.transform(df_raw[numerical_features])

        smoking_encoded = encoder.transform(df_raw[['smoking_history']])
        smoking_df = pd.DataFrame(
            smoking_encoded,
            columns=encoder.get_feature_names_out(['smoking_history'])
        )

        df_final = pd.concat([
            df_scaled[
                ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']],
            smoking_df
        ], axis=1)
        

        print(f"Final input shape: {df_final.shape}")
        print(f"Final input data:\n{df_final}")
        prediction = model.predict(df_final)
        print(f"Prediction: {prediction[0]}")

        return str(prediction[0])

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return f"Error in prediction: {str(e)}"


@app.route("/heartDiseasePrediction", methods=["POST"])
def heartDiseasePrediction():
    model = MODELS.get('hd_model')
    
    if model is None:
        return "Model for Heart Disease not available"
        
    try:
        age = int(request.form.get("age"))
        chestpain = int(request.form.get("chestpain"))
        trestbps = int(request.form.get("trestbps"))
        chol = int(request.form.get("chol"))
        restecg = int(request.form.get("restecg"))
        thalch = int(request.form.get("thalch"))
        exang = int(request.form.get("exang"))
        fbs = int(request.form.get("fbs"))
        oldpeak = float(request.form.get("oldpeak")) 
        slope = int(request.form.get("slope"))
        ca = int(request.form.get("ca"))
        thal = int(request.form.get("thal"))
        
        # Collect data for prediction
        data = [[age, chestpain, trestbps, chol, restecg, thalch, exang, fbs, oldpeak, slope, ca, thal]]
        
        # The model requires a DataFrame with specific column names
        df_predict = pd.DataFrame(data,
                                  columns=['age', 'chest pain', 'trestbps', 'chol', 'restecg', 'thalch',
                                           'exang', 'fbs', 'oldpeak', 'slope', 'ca', 'thal'])

       

        predictions = model.predict(df_predict)
        
        print(f"Heart Disease Prediction: {predictions[0]}")
        return str(predictions[0])

    except Exception as e:
        print(f"Error in heart disease prediction: {e}")
        traceback.print_exc()
        return f"Error in heart disease prediction: {str(e)}"


def load_users():
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    with open(USERS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"users": []}


def save_users(user_data):
    with open(USERS_FILE, "w") as f:
        json.dump(user_data, f, indent=4)


@app.route('/')
def select_user():
    return render_template("landingpage.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user_data = load_users()  
        users = [user for user in user_data['users'] if user['email'] == email and user['password'] == password]

        if users:
            session['userid'] = users[0]['email']
            session['username'] = users[0]['username']
            flash("Login successful!", "success")
            # Redirect to a new home page for choice
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route('/home')
def home():
    if 'username' not in session:
        flash("Please log in to access the prediction tools.", "info")
        return redirect(url_for("login"))
    return render_template("home.html", username=session.get('username'))


@app.route('/diabetes')
def diabetes():
    if 'username' not in session:
        flash("Please log in to access the prediction tools.", "info")
        return redirect(url_for("login"))
    return render_template("diabetes.html", username=session.get('username'))

@app.route('/heartdisease')
def heartdisease():
    if 'username' not in session:
        flash("Please log in to access the prediction tools.", "info")
        return redirect(url_for("login"))
    return render_template("heartdisease.html", username=session.get('username'))


@app.route('/contactUs')
def contactUs():
    name = session.get('username')
    return render_template("contactUs.html", username=name)


@app.route('/aboutUs')
def aboutUs():
    name = session.get('username')
    return render_template("aboutUs.html", username=name)

@app.route('/readMore')
def readMore():
    name = session.get('username')
    return render_template("readMore.html", username=name)

CONTACT_FILE = "contactdetails.json"

def load_json_data(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_json_data(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        contact_data = load_json_data(CONTACT_FILE)
        contact_data.append({
            'name': name,
            'email': email,
            'message': message
        })
        save_json_data(CONTACT_FILE, contact_data)

        flash("✅ Your message has been sent!", "success")
        return render_template("contactUs.html")

    return render_template("contactUs.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        age = request.form.get("age")
        email = request.form.get("email")
        password = request.form.get("password")
        user_data = load_users()
        for user in user_data["users"]:
            if user["username"] == username:
                flash("Username already exists!", "danger")
                return redirect(url_for("register"))
            if user["email"] == email:
                flash("Email already registered!", "danger")
                return redirect(url_for("register"))
        user_data["users"].append({
            "username": username,
            "age": age,
            "email": email,
            "password": password
        })
        save_users(user_data)

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()  
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    if not os.path.exists("rf.pkl"):
        print("Creating placeholder files. Remember to replace them with your actual trained models!")
        with open("rf.pkl", "wb") as f: pickle.dump("dummy_rf", f)
        with open("scaler.pkl", "wb") as f: pickle.dump("dummy_scaler", f)
        with open("encoder.pkl", "wb") as f: pickle.dump("dummy_encoder", f)
        with open("hd_model.pkl", "wb") as f: pickle.dump("dummy_hd", f)

    app.run(debug=True)
