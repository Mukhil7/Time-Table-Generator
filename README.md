# Timetable Generation Web Application

## Notice
**This project is private and not intended for public use, modification, or distribution. It is for personal or authorized use only. Unauthorized use is strictly prohibited.**

## Overview
This project is a **Timetable Generation Web Interface** built using **Streamlit**, **PyTorch**, and **Matplotlib**. It generates optimized weekly timetables for courses, teachers, rooms, and timeslots, ensuring constraints like room capacity, teacher availability, and no scheduling conflicts are met. The application features a user-friendly interface for configuring parameters and visualizing or downloading the generated timetable.

## Features
- **Configurable Parameters**: Adjust the number of courses, teachers, rooms, and timeslots via a sidebar interface.
- **Constraint-Based Scheduling**: Ensures:
  - No teacher or room is double-booked in the same timeslot.
  - Room capacities are sufficient for course sizes.
  - Predefined course-to-teacher mappings are respected.
- **Visualization**: Displays timetables as tables for each day of the week using Matplotlib.
- **CSV Export**: Download the generated timetable as a CSV file.
- **Course and Teacher Info**: Provides a summary table of courses and their assigned teachers.
- **Error Handling**: Validates input parameters to prevent out-of-bounds errors.

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: For building the web interface.
- **PyTorch**: For the neural network model that optimizes timetable assignments.
- **Matplotlib**: For rendering timetable and information tables.
- **Pandas**: For handling data and generating CSV output.
- **CUDA Support**: Utilizes GPU acceleration if available.

## Installation
*Note: Installation is restricted to authorized users only.*
1. Clone the repository (if granted access):
   ```bash
   git clone https://github.com/your-username/timetable-generation.git
   cd timetable-generation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `requirements.txt` file with:
   ```
   streamlit
   torch
   matplotlib
   pandas
   ```

## Usage
*Note: Usage is restricted to authorized users only.*
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to `http://localhost:8501`.
3. Configure the parameters in the sidebar:
   - Number of Courses (max 10)
   - Number of Teachers (max 5)
   - Number of Rooms (max 4)
   - Number of Timeslots (min 5)
4. Select a day or "All" to view the timetable.
5. Click "Generate Timetable" to train the model and display results.
6. Optionally, download the timetable as a CSV file.

## Code Structure
- **`app.py`**: Main script containing the Streamlit app, model definition, and timetable generation logic.
- **Model**: A PyTorch neural network (`TimetableModel`) that embeds courses, teachers, rooms, and timeslots to compute assignment scores.
- **Constraints**: Penalties are applied for:
  - Teacher or room conflicts in the same timeslot.
  - Room capacity violations.
- **Visualization**: Uses Matplotlib to create timetable and course-teacher information tables.
- **Data**:
  - Fixed course sizes, room capacities, and teacher assignments.
  - Configurable parameters for flexibility.

## How It Works
1. **Input Validation**: Ensures user inputs do not exceed predefined limits.
2. **Model Training**: A PyTorch model learns to assign courses to rooms and timeslots while respecting constraints.
3. **Assignment Extraction**: Greedily selects high-scoring assignments that satisfy constraints.
4. **Timetable Formatting**: Converts assignments into a visual table with time slots, including breaks and lunch.
5. **Output**: Displays timetables for selected days and provides a CSV download option.

## Example Output
- **Timetable**: A table showing course assignments (e.g., `c0(t0)`) for each room and timeslot, formatted with class, break, and lunch periods.
- **Course-Teacher Info**: A table listing course IDs, names, teacher IDs, and teacher names.
- **CSV**: A downloadable file with columns for Day, Room, Timeslot, Course, and Teacher.

## Limitations
- Fixed data for course sizes, room capacities, and teacher assignments.
- Maximum limits on courses (10), teachers (5), and rooms (4).
- Minimum of 5 timeslots to accommodate class-break-lunch structure.
- Training time may vary depending on hardware (CUDA support recommended).

## Future Improvements
- Allow dynamic course-to-teacher mappings via user input.
- Support more flexible time slot configurations (e.g., custom break/lunch durations).
- Optimize training for faster convergence.
- Add support for multi-week schedules or additional constraints (e.g., teacher preferences).

## License
This project is private and all rights are reserved. Unauthorized use, modification, or distribution is prohibited.