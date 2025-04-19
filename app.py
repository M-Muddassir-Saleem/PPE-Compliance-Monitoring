import streamlit as st
from datetime import datetime
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import random


# Initialize session state
if "ppe_df" not in st.session_state:
    st.session_state.ppe_df = None

if "description" not in st.session_state:
    st.session_state.description = ""

if "intervention" not in st.session_state:
    st.session_state.intervention = ""

if "positive_behaviour" not in st.session_state:
    st.session_state.positive_behaviour = ""

if "near_miss" not in st.session_state:
    st.session_state.near_miss = ""

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None

# Initialize session state for date, time, and location
if "date" not in st.session_state:
    st.session_state.date = datetime.now().date()  # Default to current date

if "time" not in st.session_state:
    st.session_state.time = datetime.now().time()  # Default to current time

if "location" not in st.session_state:
    st.session_state.location = "Site A"  # Default location

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from datetime import datetime
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import Spacer

def generate_pdf_report(date, time, location, ppe_df, description, intervention, positive_behaviour, near_miss, supervisor_name):
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf_filename = tmp_file.name

    # Create the PDF file
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4)

    # Create a list to hold the content of the PDF
    content = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='Title',
        fontSize=18,
        leading=22,
        alignment=1,  # Center alignment
        borderWidth=2,  # Add a 2-point border
        borderPadding=4,
        backColor=colors.darkgrey,
        borderColor=colors.black,
    )
    para_style = ParagraphStyle(
        name='Para',
        fontSize=12,
        leading=16,
        alignment=4,
        borderWidth=1,  # Add a 1-point border
        borderColor=colors.black,
        backColor=colors.lightgrey,
        borderPadding=4,
    )
    instruction = ParagraphStyle(
        name='Instruction',
        fontSize=11,
        leading=16,
        alignment=4,
        fontName='Helvetica-Oblique',
        borderWidth=1,  # Add a 1-point border
        borderColor=colors.black,
        backColor=colors.lightgrey,
        borderPadding=4,
    )
    heading_style = styles['Heading2']
    # Define a custom style for date, time, and location
    highlight_style = ParagraphStyle(
        name='Highlight',
        parent=styles['Normal'],  # Inherit from the normal style
        fontSize=14,              # Larger font size
        textColor=colors.darkblue,  # Dark blue text color
        fontName='Helvetica-Bold',  # Bold font
        spaceAfter=12,            # Add space after the paragraph
    )

    # Add title
    content.append(Paragraph("Safety Observation Card", title_style))

    # Add a spacer to create some space
    content.append(Spacer(1, 12))  # 1 inch, 12 points

    # Add instructions
    content.append(Paragraph("Instructions: The Safety Observation Card instructions involve informing the worker and completing the checklist prior to observation. Post-observation, review positive safety behaviors and discuss areas for improvement.", instruction))

    # Add a spacer to create some space
    content.append(Spacer(1, 12))  # 1 inch, 12 points

    # Add date, time, and location with emojis and custom style
    content.append(Paragraph(f"📅 <b>Date:</b> {date}", highlight_style))
    content.append(Spacer(1, 12))  # Add a spacer for better spacing

    content.append(Paragraph(f"⏰ <b>Time:</b> {time}", highlight_style))
    content.append(Spacer(1, 12))  # Add a spacer for better spacing

    content.append(Paragraph(f"📍 <b>Location:</b> {location}", highlight_style))
    content.append(Spacer(1, 12))  # Add a spacer for better spacing
    # Add a spacer to create some space
    content.append(Spacer(1, 12))  # 1 inch, 12 points

    # Add PPE checklist table
    content.append(Paragraph("Personal Protective Equipment Checklist", title_style))

    # Add a spacer to create some space
    content.append(Spacer(1, 12))  # 1 inch, 12 points

    # Convert the DataFrame to a list of lists for the table
    ppe_data_with_numbers = [["No."] + ppe_df.columns.tolist()] + [[i+1] + row for i, row in enumerate(ppe_df.values.tolist())]

    # Create the table
    ppe_table = Table(ppe_data_with_numbers, colWidths=['10%'] + ['25%'] + ['25%'] + ['*'] * (len(ppe_data_with_numbers[0]) - 4), rowHeights=[25] * len(ppe_data_with_numbers))
    ppe_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey), 
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), 
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), 
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), 
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12), 
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige), 
        ('GRID', (0, 0), (-1, -1), 1, colors.black), 
        ('FONTSIZE', (0, 0), (-1, -1), 12), 
    ]))
    content.append(ppe_table)
    # Add observation description
    content.append(Paragraph("Observation Description:", heading_style))
    content.append(Paragraph(description, para_style))

    # Add recommended interventions
    content.append(Paragraph("Recommended Interventions:", heading_style))
    content.append(Paragraph(intervention, para_style))

    # Add positive safety behaviors
    content.append(Paragraph("Positive Safety Behaviors:", heading_style))
    content.append(Paragraph(positive_behaviour, para_style))

    # Add identified near misses
    content.append(Paragraph("Identified Near Misses:", heading_style))
    content.append(Paragraph(near_miss, para_style))

    # Add supervisor name
    content.append(Paragraph("Supervisor Name:", heading_style))
    content.append(Paragraph(supervisor_name, para_style))

    # Build the PDF
    doc.build(content)

    # Return the full path of the saved PDF file
    return pdf_filename

    
# Load the YOLOv8 model
model = YOLO("my_model.pt")

# Define class colors
class_colors = {
    "hairnet": (255, 0, 0),    # Blue
    "goggles": (255, 255, 255), # White
    "mask": (0, 0, 255),        # Red
    "full-body suit": (0, 255, 0), # Green
    "gloves": (42, 42, 165),    # Brown
    "shoes": (0, 0, 0)          # Black
}

def capture_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)  # Go to last frame
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def take_random_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(num_frames):
        frame_id = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

if "current_source" not in st.session_state:
    st.session_state.current_source = "Image"

# Set page configuration
st.set_page_config(page_title="Image/Video Analysis Dashboard", layout="wide")

# Sidebar: Logo, Created by, and input fields.
with st.sidebar:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.image("stop logo.png", width=100)
    st.header("Image/Video Configuration")

    # Section 1: Select Source
    st.subheader("Select Source")
    source = st.radio(
        "Choose a source:",
        options=["Image", "Video", "Webcam"],
        index=0  # Default selection
    )

    # Section 2: Select Process
    st.subheader("Select Process")
    option = st.radio(
        "Choose an option:",
        options=["Automatic", "Manual"],
        index=0  # Default selection
    )

    # Section 3: Select Section (Multiple Selection)
    st.subheader("Select Section")
    sections = st.multiselect(
        "Choose sections to analyze:",
        options=["Head", "Face", "Eyes", "Hand", "Body", "Foot", "All"],
        default=["All"]  # Default selection
    )

    # If "All" is selected, automatically select all sections
    if "All" in sections:
        sections = ["Head", "Face", "Eyes", "Hand", "Body", "Foot"]

    # Optional: Add a "Created by" section
    st.markdown("---")  # Horizontal line for separation
    st.markdown("Created by: M Muddassir Saleem")

# Main Content Area
st.markdown(
    """<div style="text-align: center; font-size: 28px; font-weight: bold; border: 3px solid black; padding: 10px; margin-bottom: 10px;  background-color: lightgray;">Safety Observation Interface
        </div>
    """, unsafe_allow_html=True
)

# Display Frame (Image, Video, or Webcam)
st.subheader("Frame Display")

# Reset fields if the source changes
if st.session_state.current_source != source:
    st.session_state.description = ""
    st.session_state.intervention = ""
    st.session_state.positive_behaviour = ""
    st.session_state.near_miss = ""
    st.session_state.current_source = source

# Define the function at the top of the code
def generate_report_based_on_ppe_table(ppe_df, sections):
    description_list = []
    intervention_list = []
    positive_behaviour_list = []
    near_miss_list = []

    # Extract PPE status from the table
    for index, row in ppe_df.iterrows():
        ppe_item = row["Checklist"]
        section_name = ppe_item.split(" ")[0]  # Extract first word for category matching

        if section_name in sections:
            if row["Safe"] == "☑":
                positive_behaviour_list.append(ppe_item)
            elif row["Unsafe"] == "☑":
                intervention_list.append(f"{ppe_item.lower()} should be provided")
                near_miss_list.append(f"Potential risk due to missing {ppe_item.lower()}")
            else:
                description_list.append(f"{ppe_item} status unknown")

    # Convert lists to formatted sentences (professional tone)
    if description_list:
        st.session_state.description = (
            "During the site observation, workers were observed utilizing the following personal protective equipment (PPE): " 
            + ", ".join(description_list) 
            + ". However, it is essential to ensure that all required PPE is consistently worn to maintain safety standards."
        )
    else:
        st.session_state.description = (
            "All personnel were observed to be fully compliant with the required personal protective equipment (PPE) protocols. "
            "This adherence to safety standards is commendable and contributes to a safer work environment."
        )

    if intervention_list:
        st.session_state.intervention = (
            "The following interventions are required to address PPE non-compliance: " 
            + ", ".join(intervention_list) 
            + ". Immediate action is recommended to mitigate potential hazards and ensure worker safety."
        )
    else:
        st.session_state.intervention = (
            "No interventions are currently required. All personnel are adhering to the prescribed PPE protocols, "
            "which reflects a strong commitment to workplace safety."
        )

    if positive_behaviour_list:
        st.session_state.positive_behaviour = (
            "Positive safety behaviours were observed, including the proper use of the following PPE: " 
            + ", ".join(positive_behaviour_list) 
            + ". This demonstrates a proactive approach to safety and should be encouraged."
        )
    else:
        st.session_state.positive_behaviour = (
            "No specific positive behaviours related to PPE usage were recorded during this observation. "
            "It is recommended to reinforce the importance of PPE compliance through regular training and reminders."
        )

    if near_miss_list:
        st.session_state.near_miss = (
            "The following near misses were identified due to missing or improper use of PPE: " 
            + ", ".join(near_miss_list) 
            + ". These incidents highlight potential risks that could lead to injuries or accidents. "
            "Corrective measures should be implemented immediately."
        )
    else:
        st.session_state.near_miss = (
            "No near misses related to PPE usage were identified during this observation. "
            "This indicates a strong safety culture and adherence to PPE protocols."
        )
        
if source == "Image":
    if option == "Automatic":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")  # Ensure image is in RGB format
            image_np = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image", width=400)

            if st.button("Detect Objects"):
                # Process the image with YOLOv8 model
                try:
                    results = model(image_np)
                    st.write("Model inference successful!")

                    # Extract detected objects
                    detected_classes = []
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            class_name = model.names[class_id]
                            detected_classes.append(class_name)

                            # Draw bounding boxes
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            color = class_colors.get(class_name, (0, 255, 0))  # Default to green if class not found
                            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(image_np, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Display the image with bounding boxes
                    with col2:
                        st.image(image_np, caption="Detected Objects", width=400)
                        st.session_state.detect_objects_pressed = True

                    # Provide a download link for the image with bounding boxes
                    pil_image = Image.fromarray(image_np)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        pil_image.save(tmp_file.name)
                        with open(tmp_file.name, "rb") as f:
                            st.download_button(
                                label="Download Image with Bounding Boxes",
                                data=f,
                                file_name="detected_objects.png",
                                mime="image/png",
                            )

                    # STOP Card Heading
                    st.markdown(
                        """
                        <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; margin-bottom: 10px;  background-color: lightgray;">
                            Safety Observation Card
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Get the current date and time
                    current_datetime = datetime.now()
                    current_date = current_datetime.date()  # Format: YYYY-MM-DD
                    current_time = current_datetime.strftime("%I:%M %p")  # 12-hour format with AM/PM

                    # Date and Time in the same line
                    col_date, col_time = st.columns(2)
                    with col_date:
                        st.session_state.date = st.date_input("📅 Date", value=st.session_state.date)  # Use session state date
                    with col_time:
                        st.session_state.time = st.time_input("⏰ Time", value=st.session_state.time)  # Use session state time

                    # Add Location
                    st.session_state.location = st.text_input("📍 Location:", value=st.session_state.location)  # Use session state location
                    
                    st.markdown(
                    """
                    <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; background-color: lightgray; margin-bottom: 10px;"> 
                        Personal Protective Equipment Checklist
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

                    # Update the PPE table based on detected objects
                    ppe_options = ["Head Protection", "Eyes Protection", "Face Protection", "Hand Protection", "Foot Protection", "Body Protection"]
                    ppe_items = ["Hairnet", "Goggles", "Mask", "Gloves", "Shoes", "Full-body suit"]

                    table_data = {
                        "Checklist": ppe_options,
                        "PPE": ppe_items,  # Add the new PPE column
                        "N/A": ["☐"] * len(ppe_options),
                        "Safe": ["☐"] * len(ppe_options),
                        "Unsafe": ["☐"] * len(ppe_options),
                    }

                    for i, ppe in enumerate(ppe_options):
                        section_name = ppe.split(" ")[0]
                        if section_name in sections:
                            if section_name == "Head" and "hairnet" in detected_classes:
                                table_data["Safe"][i] = "☑"
                            elif section_name == "Face" and "mask" in detected_classes:
                                table_data["Safe"][i] = "☑"
                            elif section_name == "Eyes" and "goggles" in detected_classes:
                                table_data["Safe"][i] = "☑"
                            elif section_name == "Hand" and "gloves" in detected_classes:
                                table_data["Safe"][i] = "☑"
                            elif section_name == "Body" and "full-body suit" in detected_classes:
                                table_data["Safe"][i] = "☑"
                            elif section_name == "Foot" and "shoes" in detected_classes:
                                table_data["Safe"][i] = "☑"
                            else:
                                table_data["Unsafe"][i] = "☑"
                        else:
                            table_data["N/A"][i] = "☑"

                    # Convert the table data into a DataFrame for better display
                    st.session_state.ppe_df = pd.DataFrame(table_data)

                    # Add this code here
                    st.session_state.ppe_df = st.session_state.ppe_df.reset_index(drop=True)
                    st.session_state.ppe_df.index += 1
                    st.session_state.ppe_df.index.name = "No."

                    # Add color to the DataFrame
                    # st.session_state.ppe_df = st.session_state.ppe_df.style.set_properties(**{
                    #     'background-color': 'beige',
                    #     'color': 'black'
                    # })

                    st.markdown(
                        """
                        <style>
                            table {
                                width: 100%;
                                border-collapse: collapse;
                                border: 4px solid black; /* Thicker table border */
                                box-shadow: 0px 0px 10px rgba(0,0,0,0.5); /* Add shadow */
                            }
                            th, td {
                                text-align: center !important;
                                border: 3px solid black !important; /* Thicker cell borders */
                                padding: 15px !important; /* More padding */
                                font-size: 18px !important; /* Larger font size */
                                background-color: #F5F5DC; /* Light beige background */
                            }
                            th, td > span {
                                font-weight: bold; /* Bold font */
                                font-family: Arial, sans-serif; /* Font family */
                            }
                            thead th {
                                background-color: #DDD; /* Dark grey background */
                                color: white; /* White text */
                            }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Display the table
                    st.table(st.session_state.ppe_df)
                except Exception as e:
                    st.error(f"An error occurred during model inference or image processing: {e}")

    # Function to generate dynamic descriptions, interventions, positive behaviour, and near misses
    def generate_report_based_on_ppe_table(ppe_df, sections):
        description_list = []
        intervention_list = []
        positive_behaviour_list = []
        near_miss_list = []

        # Extract PPE status from the table
        for index, row in ppe_df.iterrows():
            ppe_item = row["Checklist"]
            section_name = ppe_item.split(" ")[0]  # Extract first word for category matching

            if section_name in sections:
                if row["Safe"] == "☑":
                    positive_behaviour_list.append(ppe_item)
                elif row["Unsafe"] == "☑":
                    intervention_list.append(f"{ppe_item.lower()} should be provided")
                    near_miss_list.append(f"Potential risk due to missing {ppe_item.lower()}")
                else:
                    description_list.append(f"{ppe_item} status unknown")

        # Convert lists to formatted sentences (professional tone)
        if description_list:
            st.session_state.description = (
                "During the site observation, workers were observed utilizing the following personal protective equipment (PPE): " 
                + ", ".join(description_list) 
                + ". However, it is essential to ensure that all required PPE is consistently worn to maintain safety standards."
            )
        else:
            st.session_state.description = (
                "All personnel were observed to be fully compliant with the required personal protective equipment (PPE) protocols. "
                "This adherence to safety standards is commendable and contributes to a safer work environment."
            )

        if intervention_list:
            st.session_state.intervention = (
                "The following interventions are required to address PPE non-compliance: " 
                + ", ".join(intervention_list) 
                + ". Immediate action is recommended to mitigate potential hazards and ensure worker safety."
            )
        else:
            st.session_state.intervention = (
                "No interventions are currently required. All personnel are adhering to the prescribed PPE protocols, "
                "which reflects a strong commitment to workplace safety."
            )

        if positive_behaviour_list:
            st.session_state.positive_behaviour = (
                "Positive safety behaviours were observed, including the proper use of the following PPE: " 
                + ", ".join(positive_behaviour_list) 
                + ". This demonstrates a proactive approach to safety and should be encouraged."
            )
        else:
            st.session_state.positive_behaviour = (
                "No specific positive behaviours related to PPE usage were recorded during this observation. "
                "It is recommended to reinforce the importance of PPE compliance through regular training and reminders."
            )

        if near_miss_list:
            st.session_state.near_miss = (
                "The following near misses were identified due to missing or improper use of PPE: " 
                + ", ".join(near_miss_list) 
                + ". These incidents highlight potential risks that could lead to injuries or accidents. "
                "Corrective measures should be implemented immediately."
            )
        else:
            st.session_state.near_miss = (
                "No near misses related to PPE usage were identified during this observation. "
                "This indicates a strong safety culture and adherence to PPE protocols."
            )

    # Add this after the PPE table is displayed in the "Image", "Video", or "Webcam" sections
    if st.session_state.ppe_df is not None:
        generate_report_based_on_ppe_table(st.session_state.ppe_df, sections)

    if 'detect_objects_pressed' not in st.session_state:
        st.session_state.detect_objects_pressed = False
    if source == "Image" and option == "Automatic" and st.session_state.detect_objects_pressed:

        # Display auto-generated content in a professional format
        st.markdown("**📝 Observation Description:**")
        st.text_area(
            "Detailed description of the work observed:", 
            value=st.session_state.description, 
            key="description_area",
            height=100
        )

        st.markdown("**⚠️ Recommended Interventions:**")
        st.text_area(
            "Actions required to address safety concerns:", 
            value=st.session_state.intervention, 
            key="intervention_area",
            height=100
        )

        st.markdown("**✅ Positive Safety Behaviours:**")
        st.text_area(
            "Observations of commendable safety practices:", 
            value=st.session_state.positive_behaviour, 
            key="positive_behaviour_area",
            height=100
        )

        st.markdown("**🚧 Identified Near Misses:**")
        st.text_area(
            "Potential risks or near misses observed:", 
            value=st.session_state.near_miss, 
            key="near_miss_area",
            height=100
        )

        st.markdown("**👨‍💼 Supervisor Name:**")
        supervisor_name = st.text_input(
            "Enter Supervisor Name:", 
            value="M Muddassir Saleem",  # Default value
            key="supervisor_name"
        )
        # Add a Generate Report button
        if st.button("Generate Safety Report"):
            # Generate the PDF
            st.session_state.pdf_filename = generate_pdf_report(
                date=st.session_state.date.strftime("%Y-%m-%d"),
                time=st.session_state.time.strftime("%I:%M %p"),
                location=st.session_state.location,
                ppe_df=st.session_state.ppe_df,
                description=st.session_state.description,
                intervention=st.session_state.intervention,
                positive_behaviour=st.session_state.positive_behaviour,
                near_miss=st.session_state.near_miss,
                supervisor_name=supervisor_name
            )

            # Set session state to indicate report has been generated
            if st.session_state.pdf_filename:
                st.session_state.report_generated = True

        # Check if the report has been generated and the filename is valid
        if st.session_state.report_generated and st.session_state.pdf_filename:
            st.success("Safety report generated and submitted successfully!")
            
            # Provide a download link for the PDF
            try:
                with open(st.session_state.pdf_filename, "rb") as f:
                    st.download_button(
                        label="Download Safety Observation Card (PDF)",
                        data=f,
                        file_name=os.path.basename(st.session_state.pdf_filename),  # Use the filename only
                        mime="application/pdf",
                    )
            except FileNotFoundError:
                st.error("The report file was not found. Please generate the report again.")
            except Exception as e:
                st.error(f"An error occurred while opening the report file: {e}")

            # Reset the session state (optional)
            st.session_state.report_generated = False
            st.session_state.pdf_filename = None


    # Manual Code
    elif option == "Manual":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")  # Ensure image is in RGB format
            image_np = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image", width=400)

            if st.button("Detect Objects"):
                # Process the image with YOLOv8 model
                try:
                    results = model(image_np)
                    st.write("Model inference successful!")

                    # Extract detected objects
                    detected_classes = []
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            class_name = model.names[class_id]
                            detected_classes.append(class_name)

                            # Draw bounding boxes
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            color = class_colors.get(class_name, (0, 255, 0))  # Default to green if class not found
                            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(image_np, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Display the image with bounding boxes
                    with col2:
                        st.image(image_np, caption="Detected Objects", width=400)

                    # Provide a download link for the image with bounding boxes
                    pil_image = Image.fromarray(image_np)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        pil_image.save(tmp_file.name)
                        with open(tmp_file.name, "rb") as f:
                            st.download_button(
                                label="Download Image with Bounding Boxes",
                                data=f,
                                file_name="detected_objects.png",
                                mime="image/png",
                            )
                except Exception as e:
                    st.error(f"An error occurred during model inference or image processing: {e}")

            # STOP Card Heading
            st.markdown(
                """
                <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; margin-bottom: 10px;  background-color: lightgray;">
                    Safety Observation Card
                </div>
                """,
                unsafe_allow_html=True
            )

            # Get the current date and time
            current_datetime = datetime.now()
            current_date = current_datetime.date()  # Format: YYYY-MM-DD
            current_time = current_datetime.strftime("%I:%M %p")  # 12-hour format with AM/PM

            # Date and Time in the same line
            col_date, col_time = st.columns(2)
            with col_date:
                st.session_state.date = st.date_input("📅 Date", value=current_date)
            with col_time:
                # Convert time string back to time object for st.time_input
                time_obj = datetime.strptime(current_time, "%I:%M %p").time()
                st.session_state.time = st.time_input("⏰ Time", value=time_obj)

            # Location and Work Observed
            st.session_state.location = st.text_input("📍 LOCATION", value="Site A")

            st.markdown(
                    """
                    <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; background-color: lightgray; margin-bottom: 10px;"> 
                        Personal Protective Equipment Checklist
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

            st.markdown(
                """
                <style>
                table {
                    width: 100%; /* Adjust width as needed */
                    margin: auto; /* Centers the table */
                    border-collapse: collapse;
                }
                th, td {
                    border: 2px solid black;
                    padding: 10px;
                    text-align: center !important;
                    vertical-align: middle; /* Aligns vertically in the middle */
                    background-color: beige; /* Add beige background color */
                }
                th {
                    background-color: #D3D3D3;
                    font-weight: bold;
                    text-align: center !important;
                }
                </style>

                <table>
                    <tr>
                        <th>No.</th>
                        <th>Checklist</th>
                        <th>PPE</th>
                        <th>N/A</th>
                        <th>Safe</th>
                        <th>Unsafe</th>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td>Head Protection</td>
                        <td>Hairnet</td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>Eye Protection</td>
                        <td>Goggles</td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>Face Protection</td>
                        <td>Face Mask</td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>Hand Protection</td>
                        <td>Gloves</td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                    </tr>
                    <tr>
                        <td>5</td>
                        <td>Body Protection</td>
                        <td>Full-body Suit</td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                    </tr>
                    <tr>
                        <td>6</td>
                        <td>Foot Protection</td>
                        <td>Safety Shoes</td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                        <td><input type="checkbox"></td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )


            # Description, Intervention, and Positive Behaviour
            description = st.text_area("📝 Description", height=68)  # Minimum required height
            intervention = st.text_area("⚠️ Intervention?", height=68)
            positive_behaviour_text = st.text_area("✅ Positive Behaviour", height=68)
            near_miss_text = st.text_area("🚧 Near Miss", height=68)

            # Supervisor & Worker Names
            supervisor_name = st.text_input("👨‍💼 Supervisor Name")

            # Add a Generate Report button
            if st.button("Generate Safety Report"):
                # Store manual mode fields in session state
                st.session_state.description = description
                st.session_state.intervention = intervention
                st.session_state.positive_behaviour = positive_behaviour_text
                st.session_state.near_miss = near_miss_text

                # Generate the PDF
                st.session_state.pdf_filename = generate_pdf_report(
                    date=st.session_state.date.strftime("%Y-%m-%d"),
                    time=st.session_state.time.strftime("%I:%M %p"),
                    location=st.session_state.location,
                    ppe_df=st.session_state.ppe_df,
                    description=st.session_state.description,
                    intervention=st.session_state.intervention,
                    positive_behaviour=st.session_state.positive_behaviour,
                    near_miss=st.session_state.near_miss,
                    supervisor_name=supervisor_name
                )

                # Set session state to indicate report has been generated
                if st.session_state.pdf_filename:
                    st.session_state.report_generated = True

            # Check if the report has been generated and the filename is valid
            if st.session_state.report_generated and st.session_state.pdf_filename:
                st.success("Safety report generated and submitted successfully!")
                
                # Provide a download link for the PDF
                try:
                    with open(st.session_state.pdf_filename, "rb") as f:
                        st.download_button(
                            label="Download Safety Observation Card (PDF)",
                            data=f,
                            file_name=os.path.basename(st.session_state.pdf_filename),  # Use the filename only
                            mime="application/pdf",
                        )
                except FileNotFoundError:
                    st.error("The report file was not found. Please generate the report again.")
                except Exception as e:
                    st.error(f"An error occurred while opening the report file: {e}")

                # Reset the session state (optional)
                st.session_state.report_generated = False
                st.session_state.pdf_filename = None

elif source == "Video":
    if option == "Automatic":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # Save the uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            st.video(tfile.name, format="video/mp4", start_time=0)
            
        if st.button("Take Batches from Video"):
            frames = take_random_frames(tfile.name, num_frames=8)

            if frames:
                st.session_state.frames = frames  # Store frames persistently
                st.session_state.selected_frame = 0  # Default frame selection

        if "frames" in st.session_state and st.session_state.frames:
            st.write("Select a frame for detection:")

            # Use session state to remember the selected frame
            if "selected_frame" not in st.session_state:
                st.session_state.selected_frame = 0  

            selected_frame = st.selectbox(
                "Choose a frame", 
                range(len(st.session_state.frames)), 
                format_func=lambda x: f"Frame {x+1}",
                key="selected_frame"
            )

            frame = st.session_state.frames[st.session_state.selected_frame]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_np = np.array(image)

            if image_np.shape[-1] != 3:
                st.error("The captured image must be in RGB format (3 channels).")
            else:
                # Display the selected frame and detected image in the same row
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption=f"Selected Frame {st.session_state.selected_frame+1}", width=400)

                # Add a "Detect Objects" button
                if st.button("Detect Objects"):
                    # Process the image with YOLOv8 model
                    try:
                        results = model(image_np)
                        st.write("Model inference successful!")

                        detected_classes = []
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls)
                                class_name = model.names[class_id]
                                detected_classes.append(class_name)

                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                color = class_colors.get(class_name, (0, 255, 0))
                                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(image_np, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # Display the detected image in the same row
                        with col2:
                            st.image(image_np, caption="Detected Objects", width=400)

                        # Provide a download link for the image with bounding boxes
                        pil_image = Image.fromarray(image_np)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                            pil_image.save(tmp_file.name)
                            with open(tmp_file.name, "rb") as f:
                                st.download_button(
                                    label="Download Image with Bounding Boxes",
                                    data=f,
                                    file_name="detected_objects.png",
                                    mime="image/png",
                                )

                        # STOP Card Heading
                        st.markdown(
                            """
                            <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; margin-bottom: 10px;  background-color: lightgray;">
                                Safety Observation Card
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Get the current date and time
                        current_datetime = datetime.now()
                        current_date = current_datetime.date()  # Format: YYYY-MM-DD
                        current_time = current_datetime.strftime("%I:%M %p")  # 12-hour format with AM/PM

                        # Date and Time in the same line
                        col_date, col_time = st.columns(2)
                        with col_date:
                            st.session_state.date = st.date_input("📅 Date", value=st.session_state.date)  # Use session state date
                        with col_time:
                            st.session_state.time = st.time_input("⏰ Time", value=st.session_state.time)  # Use session state time

                        # Add Location
                        st.session_state.location = st.text_input("📍 Location:", value=st.session_state.location)  # Use session state location
                        
                        st.markdown(
                        """
                        <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; background-color: lightgray; margin-bottom: 10px;"> 
                            Personal Protective Equipment Checklist
                        </div>
                        """,
                        unsafe_allow_html=True
                        )

                        # Update the PPE table based on detected objects
                        ppe_options = ["Head Protection", "Eyes Protection", "Face Protection", "Hand Protection", "Foot Protection", "Body Protection"]
                        ppe_items = ["Hairnet", "Goggles", "Mask", "Gloves", "Shoes", "Full-body suit"]

                        table_data = {
                            "Checklist": ppe_options,
                            "PPE": ppe_items,  # Add the new PPE column
                            "N/A": ["☐"] * len(ppe_options),
                            "Safe": ["☐"] * len(ppe_options),
                            "Unsafe": ["☐"] * len(ppe_options),
                        }

                        for i, ppe in enumerate(ppe_options):
                            section_name = ppe.split(" ")[0]
                            if section_name in sections:
                                if section_name == "Head" and "hairnet" in detected_classes:
                                    table_data["Safe"][i] = "☑"
                                elif section_name == "Face" and "mask" in detected_classes:
                                    table_data["Safe"][i] = "☑"
                                elif section_name == "Eyes" and "goggles" in detected_classes:
                                    table_data["Safe"][i] = "☑"
                                elif section_name == "Hand" and "gloves" in detected_classes:
                                    table_data["Safe"][i] = "☑"
                                elif section_name == "Body" and "full-body suit" in detected_classes:
                                    table_data["Safe"][i] = "☑"
                                elif section_name == "Foot" and "shoes" in detected_classes:
                                    table_data["Safe"][i] = "☑"
                                else:
                                    table_data["Unsafe"][i] = "☑"
                            else:
                                table_data["N/A"][i] = "☑"

                        # Convert the table data into a DataFrame for better display
                        st.session_state.ppe_df = pd.DataFrame(table_data)

                        # Add this code here
                        st.session_state.ppe_df = st.session_state.ppe_df.reset_index(drop=True)
                        st.session_state.ppe_df.index += 1
                        st.session_state.ppe_df.index.name = "No."

                        # Add color to the DataFrame
                        # st.session_state.ppe_df = st.session_state.ppe_df.style.set_properties(**{
                        #     'background-color': 'beige',
                        #     'color': 'black'
                        # })

                        # Apply custom CSS styles for better formatting
                        st.markdown( """
                        <style>
                        table {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        th, td {
                            text-align: center !important;
                            border: 2px solid black !important; /* Prominent table borders */
                            padding: 10px !important;
                            font-size: 16px !important;
                            background-color: beige; /* Add beige background color */
                            font-weight: bold; /* Bold headers */
                        }
                        thead th {
                            background-color: #D3D3D3; /* Light grey background */
                            color: white; /* White text */
                            font-weight: bold;
                        }
                        </style>
                        """, unsafe_allow_html=True, )

                        # Display the table
                        st.table(st.session_state.ppe_df)
                    except Exception as e:
                        st.error(f"An error occurred during model inference or image processing: {e}")

    # Function to generate dynamic descriptions, interventions, positive behaviour, and near misses
    def generate_report_based_on_ppe_table(ppe_df, sections):
        description_list = []
        intervention_list = []
        positive_behaviour_list = []
        near_miss_list = []

        # Extract PPE status from the table
        for index, row in ppe_df.iterrows():
            ppe_item = row["Checklist"]
            section_name = ppe_item.split(" ")[0]  # Extract first word for category matching

            if section_name in sections:
                if row["Safe"] == "☑":
                    positive_behaviour_list.append(ppe_item)
                elif row["Unsafe"] == "☑":
                    intervention_list.append(f"{ppe_item.lower()} should be provided")
                    near_miss_list.append(f"Potential risk due to missing {ppe_item.lower()}")
                else:
                    description_list.append(f"{ppe_item} status unknown")

        # Convert lists to formatted sentences (professional tone)
        if description_list:
            st.session_state.description = (
                "During the site observation, workers were observed utilizing the following personal protective equipment (PPE): " 
                + ", ".join(description_list) 
                + ". However, it is essential to ensure that all required PPE is consistently worn to maintain safety standards."
            )
        else:
            st.session_state.description = (
                "All personnel were observed to be fully compliant with the required personal protective equipment (PPE) protocols. "
                "This adherence to safety standards is commendable and contributes to a safer work environment."
            )

        if intervention_list:
            st.session_state.intervention = (
                "The following interventions are required to address PPE non-compliance: " 
                + ", ".join(intervention_list) 
                + ". Immediate action is recommended to mitigate potential hazards and ensure worker safety."
            )
        else:
            st.session_state.intervention = (
                "No interventions are currently required. All personnel are adhering to the prescribed PPE protocols, "
                "which reflects a strong commitment to workplace safety."
            )

        if positive_behaviour_list:
            st.session_state.positive_behaviour = (
                "Positive safety behaviours were observed, including the proper use of the following PPE: " 
                + ", ".join(positive_behaviour_list) 
                + ". This demonstrates a proactive approach to safety and should be encouraged."
            )
        else:
            st.session_state.positive_behaviour = (
                "No specific positive behaviours related to PPE usage were recorded during this observation. "
                "It is recommended to reinforce the importance of PPE compliance through regular training and reminders."
            )

        if near_miss_list:
            st.session_state.near_miss = (
                "The following near misses were identified due to missing or improper use of PPE: " 
                + ", ".join(near_miss_list) 
                + ". These incidents highlight potential risks that could lead to injuries or accidents. "
                "Corrective measures should be implemented immediately."
            )
        else:
            st.session_state.near_miss = (
                "No near misses related to PPE usage were identified during this observation. "
                "This indicates a strong safety culture and adherence to PPE protocols."
            )

    # Add this after the PPE table is displayed in the "Image", "Video", or "Webcam" sections
    if st.session_state.ppe_df is not None:
        generate_report_based_on_ppe_table(st.session_state.ppe_df, sections)

    if 'detect_objects_pressed' not in st.session_state:
        st.session_state.detect_objects_pressed = False
    if source == "Video" and option == "Automatic" and st.session_state.detect_objects_pressed:

        # Display auto-generated content in a professional format
        st.markdown("**📝 Observation Description:**")
        st.text_area(
            "Detailed description of the work observed:", 
            value=st.session_state.description, 
            key="description_area",
            height=100
        )

        st.markdown("**⚠️ Recommended Interventions:**")
        st.text_area(
            "Actions required to address safety concerns:", 
            value=st.session_state.intervention, 
            key="intervention_area",
            height=100
        )

        st.markdown("**✅ Positive Safety Behaviours:**")
        st.text_area(
            "Observations of commendable safety practices:", 
            value=st.session_state.positive_behaviour, 
            key="positive_behaviour_area",
            height=100
        )

        st.markdown("**🚧 Identified Near Misses:**")
        st.text_area(
            "Potential risks or near misses observed:", 
            value=st.session_state.near_miss, 
            key="near_miss_area",
            height=100
        )

        st.markdown("**👨‍💼 Supervisor Name:**")
        supervisor_name = st.text_input(
            "Enter Supervisor Name:", 
            value="M Muddassir Saleem",  # Default value
            key="supervisor_name"
        )
        # Add a Generate Report button
        if st.button("Generate Safety Report"):
            # Generate the PDF
            st.session_state.pdf_filename = generate_pdf_report(
                date=st.session_state.date.strftime("%Y-%m-%d"),
                time=st.session_state.time.strftime("%I:%M %p"),
                location=st.session_state.location,
                ppe_df=st.session_state.ppe_df,
                description=st.session_state.description,
                intervention=st.session_state.intervention,
                positive_behaviour=st.session_state.positive_behaviour,
                near_miss=st.session_state.near_miss,
                supervisor_name=supervisor_name
            )

            # Set session state to indicate report has been generated
            if st.session_state.pdf_filename:
                st.session_state.report_generated = True

        # Check if the report has been generated and the filename is valid
        if st.session_state.report_generated and st.session_state.pdf_filename:
            st.success("Safety report generated and submitted successfully!")
            
            # Provide a download link for the PDF
            try:
                with open(st.session_state.pdf_filename, "rb") as f:
                    st.download_button(
                        label="Download Safety Observation Card (PDF)",
                        data=f,
                        file_name=os.path.basename(st.session_state.pdf_filename),  # Use the filename only
                        mime="application/pdf",
                    )
            except FileNotFoundError:
                st.error("The report file was not found. Please generate the report again.")
            except Exception as e:
                st.error(f"An error occurred while opening the report file: {e}")

            # Reset the session state (optional)
            st.session_state.report_generated = False
            st.session_state.pdf_filename = None


    elif option == "Manual":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # Save the uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            st.video(tfile.name, format="video/mp4", start_time=0)
            
        if st.button("Take Batches from Video"):
            frames = take_random_frames(tfile.name, num_frames=8)

            if frames:
                st.session_state.frames = frames  # Store frames persistently
                st.session_state.selected_frame = 0  # Default frame selection

        if "frames" in st.session_state and st.session_state.frames:
            st.write("Select a frame for detection:")

            # Use session state to remember the selected frame
            if "selected_frame" not in st.session_state:
                st.session_state.selected_frame = 0  

            selected_frame = st.selectbox(
                "Choose a frame", 
                range(len(st.session_state.frames)), 
                format_func=lambda x: f"Frame {x+1}",
                key="selected_frame"
            )

            frame = st.session_state.frames[st.session_state.selected_frame]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_np = np.array(image)

            if image_np.shape[-1] != 3:
                st.error("The captured image must be in RGB format (3 channels).")
            else:
                # Display the selected frame and detected image in the same row
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption=f"Selected Frame {st.session_state.selected_frame+1}", width=400)

                # Add a "Detect Objects" button
                if st.button("Detect Objects"):
                    # Process the image with YOLOv8 model
                    try:
                        results = model(image_np)
                        st.write("Model inference successful!")

                        detected_classes = []
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls)
                                class_name = model.names[class_id]
                                detected_classes.append(class_name)

                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                color = class_colors.get(class_name, (0, 255, 0))
                                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(image_np, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # Display the detected image in the same row
                        with col2:
                            st.image(image_np, caption="Detected Objects", width=400)

                        # Provide a download link for the image with bounding boxes
                        pil_image = Image.fromarray(image_np)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                            pil_image.save(tmp_file.name)
                            with open(tmp_file.name, "rb") as f:
                                st.download_button(
                                    label="Download Image with Bounding Boxes",
                                    data=f,
                                    file_name="detected_objects.png",
                                    mime="image/png",
                                )

                    except Exception as e:
                        st.error(f"An error occurred during model inference or image processing: {e}")

                # STOP Card Heading
                st.markdown(
                    """
                    <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; margin-bottom: 10px;  background-color: lightgray;">
                        Safety Observation Card
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Get the current date and time
                current_datetime = datetime.now()
                current_date = current_datetime.date()  # Format: YYYY-MM-DD
                current_time = current_datetime.strftime("%I:%M %p")  # 12-hour format with AM/PM

                # Date and Time in the same line
                col_date, col_time = st.columns(2)
                with col_date:
                    date = st.date_input("📅 Date", value=current_date)
                with col_time:
                    # Convert time string back to time object for st.time_input
                    time_obj = datetime.strptime(current_time, "%I:%M %p").time()
                    time = st.time_input("⏰ Time", value=time_obj)

                # Location and Work Observed
                location = st.text_input("📍 LOCATION")

                st.markdown(
                        """
                        <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; background-color: lightgray; margin-bottom: 10px;"> 
                            Personal Protective Equipment Checklist
                        </div>
                        """,
                        unsafe_allow_html=True
                        )

                st.markdown(
                    """
                    <style>
                    table {
                        width: 100%; /* Adjust width as needed */
                        margin: auto; /* Centers the table */
                        border-collapse: collapse;
                    }
                    th, td {
                        border: 2px solid black;
                        padding: 10px;
                        text-align: center !important;
                        vertical-align: middle; /* Aligns vertically in the middle */
                        background-color: beige; /* Add beige background color */
                    }
                    th {
                        background-color: #D3D3D3;
                        font-weight: bold;
                        text-align: center !important;
                    }
                    </style>

                    <table>
                        <tr>
                            <th>No.</th>
                            <th>Checklist</th>
                            <th>PPE</th>
                            <th>N/A</th>
                            <th>Safe</th>
                            <th>Unsafe</th>
                        </tr>
                        <tr>
                            <td>1</td>
                            <td>Head Protection</td>
                            <td>Hairnet</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>2</td>
                            <td>Eye Protection</td>
                            <td>Goggles</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>3</td>
                            <td>Face Protection</td>
                            <td>Face Mask</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>4</td>
                            <td>Hand Protection</td>
                            <td>Gloves</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>5</td>
                            <td>Body Protection</td>
                            <td>Full-body Suit</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>6</td>
                            <td>Foot Protection</td>
                            <td>Safety Shoes</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )


                # Description, Intervention, and Positive Behaviour
                description = st.text_area("📝 Description", height=68)  # Minimum required height
                intervention = st.text_area("⚠️ Intervention?", height=68)
                positive_behaviour_text = st.text_area("✅ Positive Behaviour", height=68)
                near_miss_text = st.text_area("🚧 Near Miss", height=68)

                # Supervisor & Worker Names
                supervisor_name = st.text_input("👨‍💼 Supervisor Name")

            # Add a Generate Report button
            if st.button("Generate Safety Report", key="generate_report"):
                # Store manual mode fields in session state
                st.session_state.description = description
                st.session_state.intervention = intervention
                st.session_state.positive_behaviour = positive_behaviour_text
                st.session_state.near_miss = near_miss_text

                # Generate the PDF
                st.session_state.pdf_filename = generate_pdf_report(
                    date=st.session_state.date.strftime("%Y-%m-%d"),
                    time=st.session_state.time.strftime("%I:%M %p"),
                    location=st.session_state.location,
                    ppe_df=st.session_state.ppe_df,
                    description=st.session_state.description,
                    intervention=st.session_state.intervention,
                    positive_behaviour=st.session_state.positive_behaviour,
                    near_miss=st.session_state.near_miss,
                    supervisor_name=supervisor_name
                )

                # Set session state to indicate report has been generated
                if st.session_state.pdf_filename:
                    st.session_state.report_generated = True

            # Check if the report has been generated and the filename is valid
            if st.session_state.report_generated and st.session_state.pdf_filename:
                st.success("Safety report generated and submitted successfully!")
                
                # Provide a download link for the PDF
                try:
                    with open(st.session_state.pdf_filename, "rb") as f:
                        st.download_button(
                            label="Download Safety Observation Card (PDF)",
                            data=f,
                            file_name=os.path.basename(st.session_state.pdf_filename),  # Use the filename only
                            mime="application/pdf",
                        )
                except FileNotFoundError:
                    st.error("The report file was not found. Please generate the report again.")
                except Exception as e:
                    st.error(f"An error occurred while opening the report file: {e}")

                # Reset the session state (optional)
                st.session_state.report_generated = False
                st.session_state.pdf_filename = None

elif source == "Webcam":
    if option == "Automatic":
        # Initialize session state variables
        if "camera_running" not in st.session_state:
            st.session_state.camera_running = False
        if "cap" not in st.session_state:
            st.session_state.cap = None
        if "detected_classes" not in st.session_state:
            st.session_state.detected_classes = set()
        if "snapshot_taken" not in st.session_state:
            st.session_state.snapshot_taken = False
        if "snapshot" not in st.session_state:
            st.session_state.snapshot = None
        if "description" not in st.session_state:
            st.session_state.description = ""
        if "intervention" not in st.session_state:
            st.session_state.intervention = ""
        if "positive_behaviour" not in st.session_state:
            st.session_state.positive_behaviour = ""
        if "near_miss" not in st.session_state:
            st.session_state.near_miss = ""
        if "ppe_df" not in st.session_state:
            st.session_state.ppe_df = None
        if "detect_objects_pressed" not in st.session_state:
            st.session_state.detect_objects_pressed = False
        if "report_generated" not in st.session_state:
            st.session_state.report_generated = False
        if "pdf_filename" not in st.session_state:
            st.session_state.pdf_filename = None

        # Start Webcam Button
        if not st.session_state.camera_running:
            if st.button("Start Webcam"):
                st.session_state.camera_running = True
                st.session_state.cap = cv2.VideoCapture(0)  # Open webcam
                st.session_state.detected_classes = set()  # Reset detected classes
                st.session_state.snapshot_taken = False  # Reset snapshot flag
                st.session_state.snapshot = None  # Reset snapshot

        # Live Webcam Feed
        if st.session_state.camera_running:
            frame_placeholder = st.empty()  # Placeholder for live feed

            # Add a "Stop Detection" button below the live feed
            if st.button("Stop Detection"):
                st.session_state.camera_running = False
                if st.session_state.cap is not None:
                    ret, frame = st.session_state.cap.read()
                    if ret:
                        st.session_state.snapshot = frame.copy()  # Store snapshot before processing
                        st.session_state.snapshot_taken = True
                    st.session_state.cap.release()  # Release the camera
                cv2.destroyAllWindows()

            while st.session_state.camera_running:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                # Convert frame to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform YOLO inference on live frame
                results = model(frame_rgb)

                # Reset detected classes for each frame
                detected_classes = set()

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        detected_classes.add(class_name)

                        # Draw bounding boxes
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = class_colors.get(class_name, (0, 255, 0))  # Keep colors unchanged
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Update session state and table
                st.session_state.detected_classes = detected_classes
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Stop condition
                if not st.session_state.camera_running:
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                    cv2.destroyAllWindows()
                    break

        # Process the snapshot after stopping detection
        if st.session_state.snapshot_taken and st.session_state.snapshot is not None:
            # Convert snapshot to RGB for display
            original_snapshot = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_BGR2RGB)

            # Process the snapshot with YOLOv8 model
            try:
                results = model(st.session_state.snapshot)
                st.write("Model inference successful!")

                detected_classes = []
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        detected_classes.append(class_name)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = class_colors.get(class_name, (0, 255, 0))  # Keep default colors
                        cv2.rectangle(st.session_state.snapshot, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(st.session_state.snapshot, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Convert the processed image back to RGB before displaying
                processed_snapshot = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_BGR2RGB)

                # Display the captured snapshot and detected image in the same row
                col1, col2 = st.columns(2)

                with col1:
                    st.image(original_snapshot, caption="Captured Snapshot", width=400)

                with col2:
                    st.image(processed_snapshot, caption="Detected Objects", width=400)

                # STOP Card Heading
                st.markdown(
                    """
                    <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; margin-bottom: 10px;  background-color: lightgray;">
                        Observation Card
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Get the current date and time
                current_datetime = datetime.now()
                current_date = current_datetime.date()  # Format: YYYY-MM-DD
                current_time = current_datetime.strftime("%I:%M %p")  # 12-hour format with AM/PM

                # Date and Time in the same line
                col_date, col_time = st.columns(2)
                with col_date:
                    st.session_state.date = st.date_input("📅 Date", value=current_date)  # Use session state date
                with col_time:
                    st.session_state.time = st.time_input("⏰ Time", value=current_datetime.time())  # Use session state time

                # Add Location
                st.session_state.location = st.text_input("📍 Location:", value="Site Location")  # Use session state location

                # PPE Checklist
                st.markdown(
                    """
                    <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; background-color: lightgray; margin-bottom: 10px;"> 
                        Personal Protective Equipment Checklist
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Update the PPE table based on detected objects
                ppe_options = ["Head Protection", "Eyes Protection", "Face Protection", "Hand Protection", "Foot Protection", "Body Protection"]
                ppe_items = ["Hairnet", "Goggles", "Mask", "Gloves", "Shoes", "Full-body suit"]

                table_data = {
                    "Checklist": ppe_options,
                    "PPE": ppe_items,  # Add the new PPE column
                    "N/A": ["☐"] * len(ppe_options),
                    "Safe": ["☐"] * len(ppe_options),
                    "Unsafe": ["☐"] * len(ppe_options),
                }

                for i, ppe in enumerate(ppe_options):
                    section_name = ppe.split(" ")[0]
                    if section_name in sections:
                        if section_name == "Head" and "hairnet" in detected_classes:
                            table_data["Safe"][i] = "☑"
                        elif section_name == "Face" and "mask" in detected_classes:
                            table_data["Safe"][i] = "☑"
                        elif section_name == "Eyes" and "goggles" in detected_classes:
                            table_data["Safe"][i] = "☑"
                        elif section_name == "Hand" and "gloves" in detected_classes:
                            table_data["Safe"][i] = "☑"
                        elif section_name == "Body" and "full-body suit" in detected_classes:
                            table_data["Safe"][i] = "☑"
                        elif section_name == "Foot" and "shoes" in detected_classes:
                            table_data["Safe"][i] = "☑"
                        else:
                            table_data["Unsafe"][i] = "☑"
                    else:
                        table_data["N/A"][i] = "☑"

                # Convert the table data into a DataFrame for better display
                st.session_state.ppe_df = pd.DataFrame(table_data)

                # Add this code here
                st.session_state.ppe_df = st.session_state.ppe_df.reset_index(drop=True)
                st.session_state.ppe_df.index += 1
                st.session_state.ppe_df.index.name = "No."

                # Apply custom CSS styles for better formatting
                st.markdown( """
                <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    text-align: center !important;
                    border: 2px solid black !important; /* Prominent table borders */
                    padding: 10px !important;
                    font-size: 16px !important;
                    background-color: beige; /* Add beige background color */
                    font-weight: bold; /* Bold headers */
                }
                thead th {
                    background-color: #D3D3D3; /* Light grey background */
                    color: white; /* White text */
                    font-weight: bold;
                }
                </style>
                """, unsafe_allow_html=True, )

                # Display the table
                st.table(st.session_state.ppe_df)

                # Generate report based on PPE table
                generate_report_based_on_ppe_table(st.session_state.ppe_df, sections)

                # Display auto-generated content
                st.markdown("**📝 Observation Description:**")
                st.text_area("Detailed description of the work observed:", value=st.session_state.description, key="description_area", height=100)

                st.markdown("**⚠️ Recommended Interventions:**")
                st.text_area("Actions required to address safety concerns:", value=st.session_state.intervention, key="intervention_area", height=100)

                st.markdown("**✅ Positive Safety Behaviours:**")
                st.text_area("Observations of commendable safety practices:", value=st.session_state.positive_behaviour, key="positive_behaviour_area", height=100)

                st.markdown("**🚧 Identified Near Misses:**")
                st.text_area("Potential risks or near misses observed:", value=st.session_state.near_miss, key="near_miss_area", height=100)

                # Add Supervisor Name
                st.markdown("**👨‍💼 Supervisor Name:**")
                supervisor_name = st.text_input("Enter Supervisor Name:", value="M Muddassir Saleem", key="supervisor_name")

                # Add a Generate Report button
                if st.button("Generate Safety Report"):
                    # Generate the PDF
                    st.session_state.pdf_filename = generate_pdf_report(
                        date=st.session_state.date.strftime("%Y-%m-%d"),
                        time=st.session_state.time.strftime("%I:%M %p"),
                        location=st.session_state.location,
                        ppe_df=st.session_state.ppe_df,
                        description=st.session_state.description,
                        intervention=st.session_state.intervention,
                        positive_behaviour=st.session_state.positive_behaviour,
                        near_miss=st.session_state.near_miss,
                        supervisor_name=supervisor_name
                    )

                    # Set session state to indicate report has been generated
                    if st.session_state.pdf_filename:
                        st.session_state.report_generated = True

                # Check if the report has been generated and the filename is valid
                if st.session_state.report_generated and st.session_state.pdf_filename:
                    st.success("Safety report generated and submitted successfully!")

                    # Provide a download link for the PDF
                    try:
                        with open(st.session_state.pdf_filename, "rb") as f:
                            st.download_button(
                                label="Download Safety Observation Card (PDF)",
                                data=f,
                                file_name=os.path.basename(st.session_state.pdf_filename),  # Use the filename only
                                mime="application/pdf",
                            )
                    except FileNotFoundError:
                        st.error("The report file was not found. Please generate the report again.")
                    except Exception as e:
                        st.error(f"An error occurred while opening the report file: {e}")

                    # Reset the session state (optional)
                    st.session_state.report_generated = False
                    st.session_state.pdf_filename = None

            except Exception as e:
                st.error(f"An error occurred during model inference or image processing: {e}")

    elif option == "Manual":
        # Initialize session state variables
        if "camera_running" not in st.session_state:
            st.session_state.camera_running = False
        if "cap" not in st.session_state:
            st.session_state.cap = None
        if "detected_classes" not in st.session_state:
            st.session_state.detected_classes = set()
        if "snapshot_taken" not in st.session_state:
            st.session_state.snapshot_taken = False
        if "snapshot" not in st.session_state:
            st.session_state.snapshot = None

        # Start Webcam Button
        if not st.session_state.camera_running:
            if st.button("Start Webcam"):
                st.session_state.camera_running = True
                st.session_state.cap = cv2.VideoCapture(0)  # Open webcam
                st.session_state.detected_classes = set()  # Reset detected classes
                st.session_state.snapshot_taken = False  # Reset snapshot flag
                st.session_state.snapshot = None  # Reset snapshot

        # Live Webcam Feed
        if st.session_state.camera_running:
            frame_placeholder = st.empty()  # Placeholder for live feed

            # Add a "Stop Detection" button below the live feed
            if st.button("Stop Detection"):
                st.session_state.camera_running = False
                if st.session_state.cap is not None:
                    ret, frame = st.session_state.cap.read()
                    if ret:
                        st.session_state.snapshot = frame.copy()  # Store snapshot before processing
                        st.session_state.snapshot_taken = True
                    st.session_state.cap.release()  # Release the camera
                cv2.destroyAllWindows()

            while st.session_state.camera_running:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                # Convert frame to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform YOLO inference on live frame
                results = model(frame_rgb)

                # Reset detected classes for each frame
                detected_classes = set()

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        detected_classes.add(class_name)

                        # Draw bounding boxes
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = class_colors.get(class_name, (0, 255, 0))  # Keep colors unchanged
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Update session state and table
                st.session_state.detected_classes = detected_classes
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Stop condition
                if not st.session_state.camera_running:
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                    cv2.destroyAllWindows()
                    break

        # Process the snapshot after stopping detection
        if st.session_state.snapshot_taken and st.session_state.snapshot is not None:
            # Convert snapshot to RGB for display
            original_snapshot = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_BGR2RGB)

            # Process the snapshot with YOLOv8 model
            try:
                results = model(st.session_state.snapshot)
                st.write("Model inference successful!")

                detected_classes = []
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        detected_classes.append(class_name)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = class_colors.get(class_name, (0, 255, 0))  # Keep default colors
                        cv2.rectangle(st.session_state.snapshot, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(st.session_state.snapshot, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Convert the processed image back to RGB before displaying
                processed_snapshot = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_BGR2RGB)

                # Display the captured snapshot and detected image in the same row
                col1, col2 = st.columns(2)

                with col1:
                    st.image(original_snapshot, caption="Captured Snapshot", width=400)

                with col2:
                    st.image(processed_snapshot, caption="Detected Objects", width=400)

                # STOP Card Heading
                st.markdown(
                    """
                    <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; margin-bottom: 10px;  background-color: lightgray;">
                        Observation Card
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Get the current date and time
                current_datetime = datetime.now()
                current_date = current_datetime.date()  # Format: YYYY-MM-DD
                current_time = current_datetime.strftime("%I:%M %p")  # 12-hour format with AM/PM

                # Date and Time in the same line
                col_date, col_time = st.columns(2)
                with col_date:
                    date = st.date_input("📅 Date", value=current_date)
                with col_time:
                    # Convert time string back to time object for st.time_input
                    time_obj = datetime.strptime(current_time, "%I:%M %p").time()
                    time = st.time_input("⏰ Time", value=time_obj)

                # Location and Work Observed
                location = st.text_input("📍 LOCATION")

                st.markdown(
                    """
                    <div style="text-align: center; font-size: 24px; font-weight: bold; border: 3px solid black; padding: 10px; background-color: lightgray; margin-bottom: 10px;"> 
                        Personal Protective Equipment Checklist
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <style>
                    table {
                        width: 100%; /* Adjust width as needed */
                        margin: auto; /* Centers the table */
                        border-collapse: collapse;
                    }
                    th, td {
                        border: 2px solid black;
                        padding: 10px;
                        text-align: center !important;
                        vertical-align: middle; /* Aligns vertically in the middle */
                        background-color: beige; /* Add beige background color */
                    }
                    th {
                        background-color: #D3D3D3;
                        font-weight: bold;
                        text-align: center !important;
                    }
                    </style>

                    <table>
                        <tr>
                            <th>No.</th>
                            <th>Checklist</th>
                            <th>PPE</th>
                            <th>N/A</th>
                            <th>Safe</th>
                            <th>Unsafe</th>
                        </tr>
                        <tr>
                            <td>1</td>
                            <td>Head Protection</td>
                            <td>Hairnet</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>2</td>
                            <td>Eye Protection</td>
                            <td>Goggles</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>3</td>
                            <td>Face Protection</td>
                            <td>Face Mask</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>4</td>
                            <td>Hand Protection</td>
                            <td>Gloves</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>5</td>
                            <td>Body Protection</td>
                            <td>Full-body Suit</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                        <tr>
                            <td>6</td>
                            <td>Foot Protection</td>
                            <td>Safety Shoes</td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                            <td><input type="checkbox"></td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )

                # Description, Intervention, and Positive Behaviour
                description = st.text_area("📝 Description", height=68)  # Minimum required height
                intervention = st.text_area("⚠️ Intervention?", height=68)
                positive_behaviour_text = st.text_area("✅ Positive Behaviour", height=68)
                near_miss_text = st.text_area("🚧 Near Miss", height=68)

                # Supervisor & Worker Names
                supervisor_name = st.text_input("👨‍💼 Supervisor Name")

                # Add a Generate Report button
                if st.button("Generate Safety Report", key="generate_report"):
                    # Store manual mode fields in session state
                    st.session_state.description = description
                    st.session_state.intervention = intervention
                    st.session_state.positive_behaviour = positive_behaviour_text
                    st.session_state.near_miss = near_miss_text

                    # Generate the PDF
                    st.session_state.pdf_filename = generate_pdf_report(
                        date=st.session_state.date.strftime("%Y-%m-%d"),
                        time=st.session_state.time.strftime("%I:%M %p"),
                        location=st.session_state.location,
                        ppe_df=st.session_state.ppe_df,
                        description=st.session_state.description,
                        intervention=st.session_state.intervention,
                        positive_behaviour=st.session_state.positive_behaviour,
                        near_miss=st.session_state.near_miss,
                        supervisor_name=supervisor_name
                    )

                    # Set session state to indicate report has been generated
                    if st.session_state.pdf_filename:
                        st.session_state.report_generated = True

                    # Check if the report has been generated and the filename is valid
                    if st.session_state.report_generated and st.session_state.pdf_filename:
                        st.success("Safety report generated and submitted successfully!")

                        # Provide a download link for the PDF
                        try:
                            with open(st.session_state.pdf_filename, "rb") as f:
                                st.download_button(
                                    label="Download Safety Observation Card (PDF)",
                                    data=f,
                                    file_name=os.path.basename(st.session_state.pdf_filename),  # Use the filename only
                                    mime="application/pdf",
                                )
                        except FileNotFoundError:
                            st.error("The report file was not found. Please generate the report again.")
                        except Exception as e:
                            st.error(f"An error occurred while opening the report file: {e}")

                        # Reset the session state (optional)
                        st.session_state.report_generated = False
                        st.session_state.pdf_filename = None
            except Exception as e:
                st.error(f"An error occurred during model inference or image processing: {e}")