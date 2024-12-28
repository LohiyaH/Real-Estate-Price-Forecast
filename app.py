from flask import Flask, render_template, request, flash
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = "lucky"

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        try:
            vals = []
            posted_map = {'Owner': 0, 'Dealer': 1, 'Builder': 2}

            # Get form data
            posted = request.form.get("posted")
            rera = request.form.get("rera")
            rooms = request.form.get("rooms")
            square_foot = request.form.get("foot")
            ready = request.form.get("ready_to_move")
            under_construction = request.form.get("under_construction")
            location = request.form.get("location")

            # Validate form data
            if not all([posted, rera, rooms, square_foot, ready, under_construction, location]):
                flash("Please fill all the fields", "error")
                return render_template("home.html")

            # Convert values
            vals.append(posted_map[posted])  # POSTED_BY
            vals.append(int(under_construction))  # UNDER_CONSTRUCTION
            vals.append(int(rera))  # RERA
            vals.append(int(rooms))  # BHK_NO
            vals.append(float(square_foot))  # SQUARE_FT
            vals.append(int(ready))  # READY_TO_MOVE

            # Load model, scaler and label encoder
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)

            # Check if the location is in the label encoder classes
            if location in le.classes_:
                location_encoded = le.transform([location])[0]
            else:
                location_encoded = -1  # Assign a default value for unseen locations

            vals.append(location_encoded)  # ADDRESS

            # Scale the input
            scaled_vals = scaler.transform([vals])

            # Make prediction
            prediction = model.predict(scaled_vals)
            
            flash(f"The predicted price for a {rooms} BHK property in {location} is ₹{prediction[0]:.2f} Lakhs", "success")
            return render_template("home.html")

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
