import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference
import pandas as pd
import os
import numpy as np
import PyPDF2
import pymongo
from pymongo import MongoClient
from typing import Generator

# --- IBM Credentials --------
# It's good practice to keep these at the top
api_key = st.secrets["ibm"]["api_key"]
project_id = st.secrets["ibm"]["project_id"]
base_url = st.secrets["ibm"]["base_url"]
model_id = "ibm/granite-13b-instruct-v2"

# --- City Coordinates Data ---
CITY_COORDINATES = {
    "New Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Bengaluru": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Kolkata": [22.5726, 88.3639]
}

# --- Streamlit Page Config ----
st.set_page_config(
    page_title="Smart City Assistant",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Custom Styling (more modern look) ---
st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        }

        /* Chat messages */
        .stChatMessage {
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
        }
        .stChatMessage.user {
            background-color: #e0f7fa;
        }
        .stChatMessage.assistant {
            background-color: #ffffff;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            padding: 1.5rem;
        }
        
        /* Main content padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Title styling */
        h1, h2, h3 {
            color: #0d47a1;
        }
        /* Specific styling for form elements if needed */
        .stForm {
            background-color: #f0f8ff; /* Light blue background for the form */
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #4CAF50; /* Green submit button */
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Cached function to load the model ---
@st.cache_resource
def load_model():
    """Loads and caches the IBM Watsonx model."""
    try:
        # Use ModelInference as Model is deprecated
        model = ModelInference(
            model_id=model_id,
            credentials={"apikey": api_key, "url": base_url},
            project_id=project_id,
        )
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# --- CSV File Path and Saving Function for Feedback ---
# --- MongoDB Connection ---
@st.cache_resource
def get_mongo_collection():
    """Connects to MongoDB and returns the feedback collection."""
    try:
        connection_string = st.secrets["mongodb"]["connection_string"]
        client = MongoClient(connection_string)
        db_name = st.secrets["mongodb"]["database_name"]
        collection_name = st.secrets["mongodb"]["collection_name"]
        db = client[db_name]
        return db[collection_name]
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

def save_feedback_to_mongo(feedback_data: dict, collection):
    """Inserts feedback data into a MongoDB collection."""
    if collection is not None:
        try:
            collection.insert_one(feedback_data)
            return True
        except Exception as e:
            st.error(f"Failed to save feedback to MongoDB: {e}")
            return False
    return False

# --- Helper function for file processing ---
def extract_text_from_file(uploaded_file):
    """Extracts text from an uploaded file (TXT or PDF)."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    elif uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except Exception as e:
            st.error(f"Error reading PDF file '{uploaded_file.name}': {e}")
            return None
    return None


# --- Function for Summarization ---
def get_summary_response(model: ModelInference, document_text: str) -> str:
    """Generates a summary from the model."""
    if not model:
        return "Sorry, the model is not available due to a loading error."
    try:
        # A more robust prompt for summarization
        prompt = f"""
As a policy analyst assistant, summarize the following policy document into a concise, citizen-friendly version.
Focus on the key objectives, stakeholders, and outcomes. Use clear and simple language and bullet points.

Document:
---
{document_text}
---

Summary:
"""
        # generate_text returns a string directly
        response = model.generate_text(
            prompt=prompt,
            params={"max_new_tokens": 350, "temperature": 0.5, "top_p": 0.9},
        )
        return response

    except Exception as e:
        st.error(f"Error communicating with the model: {str(e)}")
        return "Sorry, I encountered an issue while trying to generate the summary."

# --- Function to run the model ---
# Encapsulating the model call in a function is good practice
def get_model_response(model: ModelInference, messages: list) -> Generator[str, None, None]:
    """Generates a streaming response from the model with conversation history. This handles 'Chat Assistant (with sample question and reply)' and 'Chat Assistant (example prompt buttons pressed)'."""
    if not model:
        yield "Sorry, the model is not available due to a loading error."
        return

# System prompt to guide the model. This has been significantly expanded for better guidance.
    system_prompt = """You are a helpful and knowledgeable Smart City Assistant. Your primary role is to provide information and advice on topics related to:
- Sustainable urban development
- Smart city technologies (e.g., IoT, AI applications in urban environments)
- Environmental policies and green infrastructure
- Urban planning, citizen engagement, and public services in a smart city context

Please adhere to the following guidelines:
1.  **Be concise and clear**: Provide direct and easy-to-understand answers.
2.  **Use bullet points**: Where appropriate, structure information using bullet points for readability.
3.  **Maintain a friendly and professional tone**: Be approachable and helpful.
4.  **Do not invent information**: If you don't know the answer or if the question is outside your defined scope, politely state that you cannot provide specific details. For example, you might say: "I am an AI assistant and cannot provide specific legal, financial, or real-time local advice. Please consult official city resources or a relevant professional for such inquiries."
5.  **Focus on general principles**: Avoid making specific policy recommendations for a particular city unless explicit context about that city's situation is provided.
6.  **Summarize complex topics**: Break down complex ideas into understandable terms for a broad audience (citizens, city planners, administrators).

Now, based on the conversation history, provide a helpful response.
"""

    # Format the conversation history for the model
    formatted_prompt = system_prompt
    for message in messages:
        role = "Input" if message["role"] == "user" else "Response"
        formatted_prompt += f"\n{role}: {message['content']}"
    formatted_prompt += "Response:"
    try:
        # Use generate_text_stream for a better user experience
        for chunk in model.generate_text_stream(
            prompt=formatted_prompt,
            params={
                "max_new_tokens": 512,
                "temperature": 0.6, # Slightly lower temperature for more focused, less creative responses
                "top_p": 0.8,      # Slightly lower top_p for more deterministic output
            }
        ):
            yield chunk
    except Exception as e:
        st.error(f"Error communicating with the model: {str(e)}")
        yield "Sorry, I encountered an issue while trying to get a response."

# --- Sidebar ---
with st.sidebar:
    st.title("🏙️ Smart City Assistant")
    st.info("Use the dashboard for city stats or chat with the AI assistant.")

    st.header("Dashboard Controls")

    # Example: Dropdown for city selection
    selected_city = st.selectbox(
        "Select a City",
        ("New Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata")
    )

    # Added a user role selector
    selected_user_role = st.selectbox(
        "Select Your Role",
        ("Citizen", "City Planner", "Administrator")
    )

    st.divider()
    st.header("Navigation")
    # This radio button group acts as the main navigation bar.
    page = st.radio(
        "Go to",
        ["🏠 Home", "📄 Policy Summarizer", "🚦 Traffic Analysis", "♻️ Waste Management", "💡 Energy Consumption", "💬 Chat Assistant", "📝 Citizen Feedback"],
        key="navigation_choice", # Add a key to control navigation
        label_visibility="collapsed"
    )

    st.info(f"Displaying dashboard for **{selected_user_role}** in **{selected_city}**.")
    st.divider()
    st.header("Chat Controls")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Example Prompts")
    example_prompts = [
        "What are the key components of a smart city?",
        "How can IoT improve waste management?",
        "Suggest some green infrastructure policies for urban areas.",
    ]
    for prompt in example_prompts:
        if st.button(prompt, use_container_width=True):
            # When an example is clicked, add it to messages and rerun
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Manually set the session state for the radio button to switch pages
            st.session_state.navigation_choice = "💬 Chat Assistant"
            st.rerun()

# --- Main Content Area ---
st.title("Smart City Intelligence Hub")

model = load_model()

# --- Page Rendering Logic ---
# Based on the sidebar navigation, we render the corresponding page.

if page == "🏠 Home":
    st.title("Welcome to the Smart City Intelligence Hub 🏙️")
    st.markdown("### Your one-stop platform for urban intelligence and sustainable planning.")
    st.markdown("""
        This application leverages the power of AI to provide insights and tools for citizens, city planners, and administrators. 
        Navigate through the different modules using the sidebar to explore various aspects of our smart city initiative.
    """)

    st.divider()

    st.header("Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Policy Summarizer")
        st.write("Upload complex policy documents and receive concise, easy-to-understand summaries. Ideal for quick reviews and public dissemination.")

        st.subheader("💬 AI Chat Assistant")
        st.write("Engage with our AI assistant for expert advice on sustainability, green infrastructure, and smart city technologies. Get your questions answered instantly.")

    with col2:
        st.subheader("📊 City Dashboards")
        st.write("Visualize key city metrics across Traffic, Waste Management, and Energy Consumption. (Note: Data is currently simulated for demonstration).")

        st.subheader("📝 Citizen Feedback")
        st.write("Report issues, provide feedback, and contribute to the improvement of your city. Your voice matters!")

    st.info("Select a page from the navigation bar on the left to get started.")

elif page == "📄 Policy Summarizer":
    st.header("Policy Document Summarization")
    st.markdown("Upload one or more policy documents (TXT or PDF) to get a concise, citizen-friendly summary.")

    uploaded_files = st.file_uploader(
        "Choose policy documents",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload plain text (.txt) or PDF (.pdf) files for analysis."
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Analysis for: `{uploaded_file.name}`")
            document_text = extract_text_from_file(uploaded_file)

            if document_text:
                with st.expander("View Original Document"):
                    st.text_area(
                        "Document Content", document_text, height=300, key=f"text_{uploaded_file.name}"
                    )

                if st.button("Summarize Document", key=f"summarize_{uploaded_file.name}"):
                    with st.spinner(f"Generating summary for {uploaded_file.name}..."):
                        summary = get_summary_response(model, document_text)
                        st.subheader("AI-Generated Summary")
                        st.markdown(summary)
            st.divider()

elif page == "🚦 Traffic Analysis":
    st.header(f"Traffic Analysis for {selected_city} ({selected_user_role} View)")
    st.info("Note: Data in this section is simulated for demonstration purposes.") # Clarifies simulated data
    # This tab handles "Traffic Analysis Tab (Map + Line Chart)"
    # Using columns to place content side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Live Traffic Map")
        # Get coordinates for the selected city
        city_coords = CITY_COORDINATES.get(selected_city, [20.5937, 78.9629]) # Default to center of India

        # Placeholder for a map visualization
        map_data = pd.DataFrame(
            np.random.randn(100, 2) / [50, 50] + city_coords,
            columns=['lat', 'lon'])
        st.map(map_data)
        st.caption("Simulated live vehicle locations.")

    with col2:
        st.subheader("Traffic Congestion Levels")
        # Use the city name to seed the random data for reproducibility per city
        # This makes the "simulated" data change when the city changes.
        seed = sum(ord(c) for c in selected_city)
        np.random.seed(seed)
        # Placeholder for a time-series chart
        chart_data = pd.DataFrame(
            np.random.rand(24, 2) * [80, 40] + [10, 20],
            columns=["Congestion Level (%)", "Average Speed (km/h)"])
        st.line_chart(chart_data)
        st.caption("Hourly traffic congestion over the last 24 hours.")

elif page == "♻️ Waste Management":
    st.header(f"Waste Management for {selected_city} ({selected_user_role} View)")
    st.info("Note: Data in this section is simulated for demonstration purposes.") # Clarifies simulated data

    # Seed the random data based on the city name for consistent-per-city-but-different-between-cities data
    seed = sum(ord(c) for c in selected_city)
    np.random.seed(seed)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recycling Rate", f"{70 + np.random.rand() * 15:.1f}%", f"{np.random.rand() * 2 - 1:.1f}%")
    col2.metric("Landfill Diversion", f"{75 + np.random.rand() * 15:.1f}%", f"{np.random.rand() * 2 - 1:.1f}%")
    col3.metric("Waste Generated (tons/day)", f"{10000 + np.random.randint(0, 5000):,}", f"{np.random.rand() * 4 - 1.5:.1f}%")

elif page == "💡 Energy Consumption":
    st.header(f"Energy Consumption for {selected_city} ({selected_user_role} View)")
    st.info("Note: Data in this section is simulated for demonstration purposes.") # Clarifies simulated data

    # Seed the random data based on the city name
    seed = sum(ord(c) for c in selected_city)
    np.random.seed(seed)

    energy_data = pd.DataFrame({
        'Hour': range(24),
        'Consumption (MWh)': np.random.rand(24) * 100 + (450 + np.random.randint(0, 100))
    })
    st.bar_chart(energy_data.set_index('Hour'))
    st.caption("Hourly energy consumption.")

elif page == "💬 Chat Assistant":
    st.header("Chat with the Assistant")
    st.markdown("Ask anything about sustainability, smart policies, green infrastructure, or city planning!")
    st.divider()

    # --- Initialize and Display Chat History ---
    # The session state is now initialized at the top of the script.

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process new user input from chat_input
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate a new response if the last message is from the user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            # Use st.write_stream to display the streaming response and get the full response
            response_generator = get_model_response(model, st.session_state.messages)
            full_response = st.write_stream(response_generator)
            # Append the full response to session state for context in the next turn
            st.session_state.messages.append({"role": "assistant", "content": full_response})

elif page == "📝 Citizen Feedback":
    st.header("Citizen Feedback & Issue Reporting")
    st.markdown("Your input helps us improve our city services. Please fill out the form below to report an issue or provide feedback.") # Handles "Citizen Feedback Tab (form empty)" initially


    with st.form("citizen_feedback_form", clear_on_submit=True):
        st.subheader("Report an Issue or Provide Feedback")
        
        name = st.text_input("Your Name (Optional)", key="feedback_name")
        contact = st.text_input("Your Email or Phone (Optional)", key="feedback_contact")
        
        category = st.selectbox(
            "Category of Feedback/Issue",
            ["General Feedback", "Water & Sanitation", "Roads & Infrastructure", "Environment & Green Spaces", "Public Safety", "Waste Management", "Energy", "Other"],
            key="feedback_category"
        )
        
        location = st.text_input("Location of Issue (e.g., Street Name, Landmark, Intersection)", key="feedback_location")
        
        description = st.text_area(
            "Detailed Description of Feedback/Issue",
            help="Please provide as much detail as possible, including dates, times, and specific observations.",
            key="feedback_description"
        )
        
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if not description or len(description) < 10:
                st.error("Please provide a detailed description (at least 10 characters) of your feedback or issue.")
            else:
                feedback_data = {
                    "Timestamp": pd.Timestamp.now(),
                    "Name": name if name else "Anonymous",
                    "Contact": contact if contact else "N/A",
                    "Category": category,
                    "Location": location if location else "N/A",
                    "Description": description
                }
                collection = get_mongo_collection()
                if save_feedback_to_mongo(feedback_data, collection):
                    st.success("Thank you for your feedback! Your submission has been recorded.")
