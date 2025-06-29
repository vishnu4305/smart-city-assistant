# Smart City Assistant 🏙️

A comprehensive, AI-powered Streamlit dashboard designed to provide insights and tools for urban intelligence and sustainable city planning.

## Overview

The Smart City Assistant is an interactive web application that serves as a central hub for various smart city initiatives. It leverages the power of IBM's Watsonx AI to offer several key functionalities, from answering complex policy questions to summarizing documents and collecting citizen feedback. The application is designed for multiple user roles, including citizens, city planners, and administrators, providing tailored views and tools for each.

## ✨ Key Features

*   **🤖 AI Chat Assistant**: Engage in a conversation with an AI powered by IBM Watsonx. Get instant answers on topics like sustainable development, green infrastructure, and smart city technologies.
*   **📄 Policy Summarizer**: Upload dense policy documents (in `.txt` or `.pdf` format) and receive concise, easy-to-understand summaries, making complex information accessible to everyone.
*   **📊 Interactive Dashboards**: Visualize key city metrics with simulated data for:
    *   **Traffic Analysis**: View live traffic maps and congestion level charts.
    *   **Waste Management**: Track recycling rates and landfill diversion metrics.
    *   **Energy Consumption**: Analyze hourly energy usage patterns.
*   **📝 Citizen Feedback System**: A dedicated portal for citizens to report issues, submit feedback, and contribute to urban improvement. All feedback is stored securely in a MongoDB database.
*   **👤 Role-Based Views**: The dashboard experience is tailored based on the selected user role (Citizen, City Planner, Administrator).

---
## 📸 Screenshots

*(Placeholder for screenshots of the application)*

**Home Page**
!Home Page

**Chat Assistant**
!Chat Assistant

**Policy Summarizer**
!Policy Summarizer

---
## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **AI / Large Language Model**: IBM Watsonx AI (`ibm/granite-13b-instruct-v2`)
*   **Database**: MongoDB
*   **Data Processing**: Pandas, NumPy
*   **PDF Parsing**: PyPDF2

---
## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

*   Python 3.8+
*   Git

### 1. Clone the Repository

```bash
git clone https://github.com/vishnu4305/smart-city-assistant.git
cd smart-city-assistant
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Credentials

The application uses Streamlit's secrets management. Create a file named `.streamlit/secrets.toml` in the root of your project directory and add your credentials in the following format:

```toml
# .streamlit/secrets.toml

[ibm]
api_key = "YOUR_IBM_WATSONX_API_KEY"
project_id = "YOUR_IBM_WATSONX_PROJECT_ID"
base_url = "https://us-south.ml.cloud.ibm.com" # Or your specific region

[mongodb]
connection_string = "YOUR_MONGODB_CONNECTION_STRING"
database_name = "YOUR_DATABASE_NAME"
collection_name = "YOUR_COLLECTION_NAME"
```
**Important**: Do not commit your `secrets.toml` file to version control. The `.gitignore` file should already be configured to ignore it.

### 5. Run the Application

Once the dependencies are installed and secrets are configured, you can run the Streamlit app:

```bash
streamlit run app.py
```

The application should now be running and accessible in your web browser!

---
## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, please feel free to:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---
## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
