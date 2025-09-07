import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Validate input parameters to prevent out-of-bounds errors
def validate_inputs(NUM_COURSES, NUM_TEACHERS, NUM_ROOMS, NUM_TIMESLOTS):
    max_rooms = 4  # Length of ROOM_CAPACITIES and ROOM_NAMES
    max_courses = 10  # Length of COURSE_SIZES and COURSE_NAMES
    max_teachers = 5  # Length of TEACHER_NAMES
    min_timeslots = 5  # Minimum to accommodate class-break-lunch structure

    if NUM_ROOMS > max_rooms:
        raise ValueError(
            f"NUM_ROOMS ({NUM_ROOMS}) exceeds maximum allowed value. "
            f"Maximum: {max_rooms}, Minimum: 1"
        )
    if NUM_COURSES > max_courses:
        raise ValueError(
            f"NUM_COURSES ({NUM_COURSES}) exceeds maximum allowed value. "
            f"Maximum: {max_courses}, Minimum: 1"
        )
    if NUM_TEACHERS > max_teachers:
        raise ValueError(
            f"NUM_TEACHERS ({NUM_TEACHERS}) exceeds maximum allowed value. "
            f"Maximum: {max_teachers}, Minimum: 1"
        )
    if NUM_TIMESLOTS < min_timeslots:
        raise ValueError(
            f"NUM_TIMESLOTS ({NUM_TIMESLOTS}) is below minimum allowed value. "
            f"Minimum: {min_timeslots}, Maximum: No strict upper limit"
        )

# Configurable variables
st.sidebar.header("Configuration")
NUM_COURSES = st.sidebar.number_input("Number of Courses", min_value=1, value=10)
NUM_TEACHERS = st.sidebar.number_input("Number of Teachers", min_value=1, value=5)
NUM_ROOMS = st.sidebar.number_input("Number of Rooms", min_value=1, value=4)
NUM_TIMESLOTS = st.sidebar.number_input("Number of Timeslots", min_value=1, value=7)
CLASS_DURATION = 55
BREAK_DURATION = 20
LUNCH_DURATION = 50
START_TIME = 8 * 60 + 55
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ROOM_CAPACITIES = torch.tensor([30, 50, 40, 20], dtype=torch.float32)[:NUM_ROOMS]
ROOM_NAMES = ["1st Year", "2nd Year", "3rd Year", "4th Year"][:NUM_ROOMS]

# Fixed data
COURSE_SIZES = torch.tensor([25, 35, 15, 20, 30, 10, 40, 25, 20, 15], dtype=torch.float32)[:NUM_COURSES]
COURSE_NAMES = ["AI Fundamentals", "ML2", "Cloud Computing", "Web Programming",
                "Database Systems", "Algorithms", "Data Science", "Operating Systems",
                "Software Engineering", "Computer Network"][:NUM_COURSES]
TEACHER_NAMES = ["Anandha Krishnan", "Anish Antomy", "Karthick", "Biplap Das", "Kirthika"][:NUM_TEACHERS]

# Manual course-to-teacher mapping
COURSE_TEACHER_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
COURSE_TEACHER_MAP = {k: v for k, v in COURSE_TEACHER_MAP.items() if k < NUM_COURSES and v < NUM_TEACHERS}

# Validate inputs
try:
    validate_inputs(NUM_COURSES, NUM_TEACHERS, NUM_ROOMS, NUM_TIMESLOTS)
except ValueError as e:
    st.error(str(e))
    st.stop()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOM_CAPACITIES = ROOM_CAPACITIES.to(device)
COURSE_SIZES = COURSE_SIZES.to(device)
COURSE_TEACHER = torch.tensor([COURSE_TEACHER_MAP.get(i, 0) for i in range(NUM_COURSES)], dtype=torch.long).to(device)

# Model
class TimetableModel(nn.Module):
    def __init__(self, n_courses, n_teachers, n_rooms, n_timeslots, embed_dim=16):
        super().__init__()
        self.course_emb = nn.Embedding(n_courses, embed_dim)
        self.teacher_emb = nn.Embedding(n_teachers, embed_dim)
        self.room_emb = nn.Embedding(n_rooms, embed_dim)
        self.timeslot_emb = nn.Embedding(n_timeslots, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, course_idx, teacher_idx, room_idx, timeslot_idx):
        invalid_mask = (teacher_idx != COURSE_TEACHER[course_idx].to(device))
        x = torch.cat([
            self.course_emb(course_idx),
            self.teacher_emb(teacher_idx),
            self.room_emb(room_idx),
            self.timeslot_emb(timeslot_idx)], dim=-1)
        x = torch.relu(self.fc1(x))
        score = self.fc2(x).squeeze(-1)
        score[invalid_mask] = -1e9
        return score

def generate_all_assignments(n_courses, n_rooms, n_timeslots):
    c = torch.arange(n_courses).unsqueeze(1).unsqueeze(2)
    r = torch.arange(n_rooms).unsqueeze(0).unsqueeze(2)
    t = torch.arange(n_timeslots).unsqueeze(0).unsqueeze(1)
    c = c.expand(n_courses, n_rooms, n_timeslots).reshape(-1)
    r = r.expand(n_courses, n_rooms, n_timeslots).reshape(-1)
    t = t.expand(n_courses, n_rooms, n_timeslots).reshape(-1)
    return c.to(device), r.to(device), t.to(device)

def constraint_penalties(assignments, course_teacher, room_caps, course_sizes, n_teachers, n_timeslots, n_rooms):
    courses, rooms, timeslots = assignments
    penalty = 0.0
    teacher_ids = course_teacher[courses]
    for ts in range(n_timeslots):
        mask = (timeslots == ts)
        _, counts = torch.unique(teacher_ids[mask], return_counts=True)
        penalty += (counts - 1).clamp(min=0).sum() * 100
        _, rcounts = torch.unique(rooms[mask], return_counts=True)
        penalty += (rcounts - 1).clamp(min=0).sum() * 100
    penalty += (course_sizes[courses] - room_caps[rooms]).clamp(min=0).sum() * 10
    return penalty

def format_timetable(assignments, day):
    time_slots = []
    current = START_TIME
    total_periods = NUM_TIMESLOTS
    periods_before_break = 2
    periods_after_break = 2
    periods_after_lunch = total_periods - (periods_before_break + periods_after_break)

    for i in range(periods_before_break):
        start, end = current, current + CLASS_DURATION
        time_slots.append((f"{start//60}:{start%60:02d}-{end//60}:{end%60:02d}", "Class"))
        current = end
    start, end = current, current + BREAK_DURATION
    time_slots.append((f"{start//60}:{start%60:02d}-{end//60}:{end%60:02d}", "Break"))
    current = end
    for i in range(periods_after_break):
        start, end = current, current + CLASS_DURATION
        time_slots.append((f"{start//60}:{start%60:02d}-{end//60}:{end%60:02d}", "Class"))
        current = end
    start, end = current, current + LUNCH_DURATION
    time_slots.append((f"{start//60}:{start%60:02d}-{end//60}:{end%60:02d}", "Lunch"))
    current = end
    for i in range(periods_after_lunch):
        start, end = current, current + CLASS_DURATION
        time_slots.append((f"{start//60}:{start%60:02d}-{end//60}:{end%60:02d}", "Class"))
        current = end

    timetable = [["-" for _ in range(len(time_slots))] for _ in range(NUM_ROOMS)]
    class_idx = [i for i, (tr, st) in enumerate(time_slots) if st == "Class"]
    for c, r, t in assignments:
        if t < len(class_idx):
            idx = class_idx[t]
            timetable[r][idx] = f"c{c}(t{COURSE_TEACHER[c].item()})"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    col_labels = ["Room"] + [f"{tr} ({st})" for tr, st in time_slots]
    data = [[ROOM_NAMES[r]] + timetable[r] for r in range(NUM_ROOMS)]
    table = ax.table(cellText=data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.title(f"Weekly Timetable ({day})", fontsize=12, pad=15)
    return fig

def extract_assignments(model, courses, rooms, timeslots, course_teacher, nc, nr, nt):
    per_day = {}
    used = {t: set() for t in range(NUM_TEACHERS)}
    for day in DAYS:
        slots = {t: set() for t in range(nt)}
        teach = {t: set() for t in range(nt)}
        assigned = set()
        chosen = []
        with torch.no_grad():
            scores = model(courses, course_teacher[courses], rooms, timeslots)
        _, idxs = torch.sort(scores, descending=True)
        for i in idxs:
            c = courses[i].item()
            r = rooms[i].item()
            t = timeslots[i].item()
            teacher = course_teacher[c].item()
            if (c not in assigned and
                r not in slots[t] and
                teacher not in teach[t] and
                COURSE_SIZES[c] <= ROOM_CAPACITIES[r] and
                (c, t) not in used[teacher]):
                chosen.append((c, r, t))
                assigned.add(c)
                slots[t].add(r)
                teach[t].add(teacher)
                used[teacher].add((c, t))
                if len(assigned) == nc:
                    break
        per_day[day] = chosen
    return per_day

def timetable_to_csv(assign):
    data = []
    for day, assignments in assign.items():
        for c, r, t in assignments:
            data.append([day, ROOM_NAMES[r], t, f"c{c}(t{COURSE_TEACHER[c].item()})", TEACHER_NAMES[COURSE_TEACHER[c].item()]])
    df = pd.DataFrame(data, columns=["Day", "Room", "Timeslot", "Course", "Teacher"])
    return df.to_csv(index=False)

@st.cache_resource
def train_timetable():
    model = TimetableModel(NUM_COURSES, NUM_TEACHERS, NUM_ROOMS, NUM_TIMESLOTS).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    courses, rooms, timeslots = generate_all_assignments(NUM_COURSES, NUM_ROOMS, NUM_TIMESLOTS)
    progress = st.progress(0)
    for epoch in range(200):  # Reduced for faster local testing
        model.train()
        opt.zero_grad()
        scores = model(courses, COURSE_TEACHER[courses], rooms, timeslots)
        probs = torch.softmax(scores, dim=0)
        penalty = constraint_penalties((courses, rooms, timeslots), COURSE_TEACHER, ROOM_CAPACITIES, COURSE_SIZES, NUM_TEACHERS, NUM_TIMESLOTS, NUM_ROOMS)
        loss = penalty + 0.1 * (-torch.sum(probs * scores))
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            progress.progress((epoch + 20) / 200)
            st.write(f"Epoch {epoch} | Loss: {loss.item():.2f}")
    assign = extract_assignments(model, courses, rooms, timeslots, COURSE_TEACHER, NUM_COURSES, NUM_ROOMS, NUM_TIMESLOTS)
    return assign

st.title("Timetable Generation Web Interface")
st.write("Configure the timetable parameters in the sidebar and select a day to view or generate the full timetable.")

selected_day = st.selectbox("Select Day to View", ["All"] + DAYS)

if st.button("Generate Timetable"):
    with st.spinner("Training model and generating timetable..."):
        try:
            assign = train_timetable()
            if selected_day == "All":
                for d in DAYS:
                    fig = format_timetable(assign[d], d)
                    st.pyplot(fig)
            else:
                fig = format_timetable(assign[selected_day], selected_day)
                st.pyplot(fig)

            # Course/teacher info
            info = [["Course", "Name", "Teacher", "Name"]]
            for i in range(NUM_COURSES):
                teacher_idx = COURSE_TEACHER_MAP.get(i, 0)
                info.append([f"c{i}", COURSE_NAMES[i], f"t{teacher_idx}", TEACHER_NAMES[teacher_idx]])
            fig, ax = plt.subplots(figsize=(6, NUM_COURSES * 0.6 + 1))
            ax.axis('off')
            table = ax.table(cellText=info, colLabels=info[0], cellLoc='center', loc='center',
                             colColours=['#f0f0f0'] * len(info[0]),
                             cellColours=[['#ffffff'] * len(info[0]) for _ in range(NUM_COURSES + 1)])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.4, 1.4)
            plt.title("Course and Teacher Information", pad=20)
            st.pyplot(fig)

            # Download CSV
            csv_data = timetable_to_csv(assign)
            st.download_button("Download Timetable as CSV", csv_data, "timetable.csv", "text/csv")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")