import os
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np  # Add this line


import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Database connection function
def get_db_connection():
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 18 for SQL Server};'
            'SERVER=Suprith.database.windows.net;'
            'DATABASE=Suprith;'
            'UID=kollapvn;'
            'PWD=Cherry@1718;'
            'Encrypt=yes;'
            'TrustServerCertificate=yes;'
        )
        return conn
    except pyodbc.Error as e:
        print(f"Error connecting to database: {e}")
        return None


@app.route('/sample_data_pull')
def sample_data_pull():
    # Pull data for HH #10
    hshd_num = 10
    rows = None
    error_message = None

    # Query the database for the data related to HSHD_NUM = 10
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        query = """
        SELECT
            H.HSHD_NUM,
            T.BASKET_NUM,
            T.DATE,
            P.PRODUCT_NUM,
            P.DEPARTMENT,
            P.COMMODITY
        FROM
            Households H
        JOIN
            Transactions T ON H.HSHD_NUM = T.HSHD_NUM
        JOIN
            Products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
        WHERE
            H.HSHD_NUM = ?
        ORDER BY
            H.HSHD_NUM, T.BASKET_NUM, T.DATE, P.PRODUCT_NUM, P.DEPARTMENT, P.COMMODITY
        """
        try:
            cursor.execute(query, (hshd_num,))
            rows = cursor.fetchall()
        except Exception as e:
            error_message = f"Error executing the query: {str(e)}"
        finally:
            cursor.close()
            conn.close()
    
    if not rows:
        error_message = f"No data found for Household Number: {hshd_num}"

    # Render the template with the data
    return render_template('sample_datapull.html', rows=rows, hshd_num=hshd_num, error_message=error_message)


@app.route('/search', methods=['GET', 'POST'])
def search():
    hshd_num = None
    rows = None
    error_message = None

    if request.method == 'POST':
        hshd_num = request.form.get('hshd_num')
        if not hshd_num or not hshd_num.isdigit():
            error_message = "Invalid input. Please enter a valid Household Number."
        else:
            hshd_num = int(hshd_num)
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                query = """
                SELECT
                    H.HSHD_NUM,
                    T.BASKET_NUM,
                    T.DATE,
                    P.PRODUCT_NUM,
                    P.DEPARTMENT,
                    P.COMMODITY
                FROM
                    Households H
                JOIN
                    Transactions T ON H.HSHD_NUM = T.HSHD_NUM
                JOIN
                    Products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
                WHERE
                    H.HSHD_NUM = ?
                ORDER BY
                    T.DATE, T.BASKET_NUM
                """
                try:
                    cursor.execute(query, (hshd_num,))
                    rows = cursor.fetchall()
                except Exception as e:
                    error_message = f"Error executing the query: {str(e)}"
                finally:
                    cursor.close()
                    conn.close()
            else:
                error_message = "Error: Unable to connect to the database."

    return render_template('search.html', rows=rows, hshd_num=hshd_num, error_message=error_message)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        transactions_file = request.files.get('transactions')
        households_file = request.files.get('households')
        products_file = request.files.get('products')

        if not (transactions_file and households_file and products_file):
            flash("Please upload all three files: Transactions, Households, and Products.", "error")
            return redirect(request.url)

        try:
            transactions_df = pd.read_csv(transactions_file)
            households_df = pd.read_csv(households_file)
            products_df = pd.read_csv(products_file)
        except Exception as e:
            flash(f"Error reading CSV files: {str(e)}", "error")
            return redirect(request.url)

        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                # Upload Transactions
                cursor.execute("TRUNCATE TABLE Transactions")
                for _, row in transactions_df.iterrows():
                    cursor.execute(
                        "INSERT INTO Transactions (HSHD_NUM, BASKET_NUM, DATE, PRODUCT_NUM, SPEND) VALUES (?, ?, ?, ?, ?)",
                        row['HSHD_NUM'], row['BASKET_NUM'], row['DATE'], row['PRODUCT_NUM'], row['SPEND']
                    )
                # Upload Households
                cursor.execute("TRUNCATE TABLE Households")
                for _, row in households_df.iterrows():
                    cursor.execute(
                        "INSERT INTO Households (HSHD_NUM, AGE_RANGE, MARITAL_STATUS, INCOME_RANGE) VALUES (?, ?, ?, ?)",
                        row['HSHD_NUM'], row['AGE_RANGE'], row['MARITAL_STATUS'], row['INCOME_RANGE']
                    )
                # Upload Products
                cursor.execute("TRUNCATE TABLE Products")
                for _, row in products_df.iterrows():
                    cursor.execute(
                        "INSERT INTO Products (PRODUCT_NUM, DEPARTMENT, COMMODITY, BRAND_TY) VALUES (?, ?, ?, ?)",
                        row['PRODUCT_NUM'], row['DEPARTMENT'], row['COMMODITY'], row['BRAND_TY']
                    )
                conn.commit()
                flash("Data uploaded successfully!", "success")
            except Exception as e:
                flash(f"Error uploading data: {str(e)}", "error")
            finally:
                conn.close()
        else:
            flash("Error: Unable to connect to the database.", "error")

        return redirect(url_for('upload'))

    return render_template('upload_data.html')


@app.route('/dashboard')
def dashboard():
    conn = get_db_connection()
    if conn is None:
        return "Error: Unable to connect to the database."

    cursor = conn.cursor()

    # Queries
    query_demographics = """
        SELECT AGE_RANGE, INCOME_RANGE, COUNT(*) AS NUM_HOUSEHOLDS
        FROM Households
        GROUP BY AGE_RANGE, INCOME_RANGE
    """
    query_engagement = """
        SELECT YEAR(T.DATE) AS YEAR, MONTH(T.DATE) AS MONTH, SUM(T.SPEND) AS TOTAL_SPENDING
        FROM Transactions T
        GROUP BY YEAR(T.DATE), MONTH(T.DATE)
        ORDER BY YEAR(T.DATE), MONTH(T.DATE)
    """
    query_basket = """
        SELECT BASKET_NUM, COUNT(DISTINCT PRODUCT_NUM) AS NUM_PRODUCTS
        FROM Transactions
        GROUP BY BASKET_NUM
    """
    query_seasonal = """
        SELECT MONTH(T.DATE) AS MONTH, SUM(T.SPEND) AS TOTAL_SPENDING
        FROM Transactions T
        GROUP BY MONTH(T.DATE)
        ORDER BY MONTH(T.DATE)
    """
    query_brand_preferences = """
        SELECT P.Brand_type, SUM(T.SPEND) AS TOTAL_SPENDING
        FROM Transactions T
        JOIN Products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
        GROUP BY P.Brand_type
    """

    try:
        # Fetch data for each section
        cursor.execute(query_demographics)
        demographics_data = cursor.fetchall()

        cursor.execute(query_engagement)
        engagement_data = cursor.fetchall()

        cursor.execute(query_basket)
        basket_data = cursor.fetchall()

        cursor.execute(query_seasonal)
        seasonal_data = cursor.fetchall()

        cursor.execute(query_brand_preferences)
        brand_data = cursor.fetchall()
    except Exception as e:
        return f"Error executing the queries: {str(e)}"
    finally:
        conn.close()

    # Visualizations
    # Engagement Over Time
    engagement_years_months = [f"{row[0]}-{row[1]:02d}" for row in engagement_data]
    engagement_spending = [row[2] for row in engagement_data]
    plt.figure(figsize=(10, 6))
    plt.plot(engagement_years_months, engagement_spending, marker='o', linestyle='-', color='blue')
    plt.title('Engagement Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Spending')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    engagement_img = BytesIO()
    plt.savefig(engagement_img, format='png')
    engagement_img.seek(0)
    engagement_base64 = base64.b64encode(engagement_img.getvalue()).decode('utf-8')
    plt.close()

    # Seasonal Trends
    seasonal_months = [row[0] for row in seasonal_data]
    seasonal_spending = [row[1] for row in seasonal_data]
    plt.figure(figsize=(8, 5))
    plt.bar(seasonal_months, seasonal_spending, color='skyblue')
    plt.title('Seasonal Spending Trends')
    plt.xlabel('Month')
    plt.ylabel('Total Spending')
    plt.tight_layout()
    seasonal_img = BytesIO()
    plt.savefig(seasonal_img, format='png')
    seasonal_img.seek(0)
    seasonal_base64 = base64.b64encode(seasonal_img.getvalue()).decode('utf-8')
    plt.close()

    # Brand Preferences
    brand_types = [row[0] for row in brand_data]
    brand_spending = [row[1] for row in brand_data]
    plt.figure(figsize=(8, 5))
    plt.pie(brand_spending, labels=brand_types, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightblue', 'salmon'])
    plt.title('Brand Preferences')
    plt.tight_layout()
    brand_img = BytesIO()
    plt.savefig(brand_img, format='png')
    brand_img.seek(0)
    brand_base64 = base64.b64encode(brand_img.getvalue()).decode('utf-8')
    plt.close()

    return render_template(
        'dashboard.html',
        demographics_data=demographics_data,
        engagement_base64=engagement_base64,
        basket_data=basket_data,
        seasonal_base64=seasonal_base64,
        brand_base64=brand_base64
    )

@app.route('/basket_analysis', methods=['GET', 'POST'])
def basket_analysis():
    if request.method == 'POST':
        try:
            # Step 1: Load data from uploaded CSVs
            transactions_file = request.files['transactions']
            products_file = request.files['products']
            households_file = request.files['households']

            transactions = pd.read_csv(transactions_file)
            products = pd.read_csv(products_file)
            households = pd.read_csv(households_file)

            # Step 2: Merge data based on common keys
            merged_data = transactions.merge(products, on="PRODUCT_NUM").merge(households, on="HSHD_NUM")

            # Step 3: Preprocess for Basket Analysis
            basket_data = merged_data.groupby('BASKET_NUM')['PRODUCT_NUM'].apply(list).reset_index()

            # One-hot encode the products
            mlb = MultiLabelBinarizer()
            basket_encoded = pd.DataFrame(mlb.fit_transform(basket_data['PRODUCT_NUM']),
                                          columns=mlb.classes_,
                                          index=basket_data['BASKET_NUM'])

            # Step 4: Prepare target variable (e.g., total basket spend or basket size)
            basket_data['TARGET'] = basket_encoded.sum(axis=1)  # Basket size (example)

            # Step 5: Train the Random Forest Model
            X = basket_encoded
            y = basket_data['TARGET']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Step 6: Evaluate Model
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Step 7: Get Feature Importance (for cross-selling opportunities)
            feature_importances = pd.DataFrame({
                'Product': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            top_products_html = feature_importances.head(10).to_html(index=False)

            return render_template('basket_analysis.html',
                                   accuracy=accuracy,
                                   report=report,
                                   top_products_html=top_products_html)
        except Exception as e:
            flash(f"Error during basket analysis: {str(e)}", "error")
            return redirect(url_for('basket_analysis'))

    return render_template('upload_basket_data.html')



# Step 1: Create and Train the Model (this could also be done offline and loaded as a pickle file)
def train_model():
    # Example synthetic data (replace with your data loading logic)
    np.random.seed(42)
    data = pd.DataFrame({
        'Recency': np.random.randint(1, 365, 1000),
        'Frequency': np.random.randint(1, 50, 1000),
        'Monetary': np.random.uniform(10, 1000, 1000),
        'EngagementScore': np.random.uniform(0, 1, 1000),
        'Churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })

    X = data[['Recency', 'Frequency', 'Monetary', 'EngagementScore']]
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

# Step 2: Define Routes
@app.route('/churn')
def churnhome():
    return "Welcome to the Churn Prediction App!"

from flask import Flask, request, jsonify

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data from the POST request
        data = request.get_json()  # Correct way to fetch JSON data
        recency = data['Recency']
        frequency = data['Frequency']
        monetary = data['Monetary']
        engagement_score = data['EngagementScore']

        # Prepare the features for prediction
        input_data = np.array([[recency, frequency, monetary, engagement_score]])

        # Predict using the trained model
        prediction = model.predict(input_data)
        churn_probability = model.predict_proba(input_data)[0][1]  # Get probability for churn (class 1)

        return jsonify({
            "prediction": int(prediction[0]),
            "churn_probability": round(churn_probability, 2)
        })

    except KeyError as e:
        return jsonify({"error": f"Missing required input: {str(e)}"}), 400



@app.route('/')
def home():
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True)