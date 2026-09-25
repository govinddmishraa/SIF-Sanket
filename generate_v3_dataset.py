import pandas as pd
import random

random.seed(2026)

OUTPUT_FILE = "dataset/sif_sanket_synthetic_safety_reports_v3.csv"

reports = []

# =========================================================
# BASIC DATA
# =========================================================

sites = [
    "Refinery Area A",
    "Gas Processing Unit",
    "Pipeline Station 3",
    "Compressor Station 2",
    "Tank Farm",
    "Maintenance Workshop",
    "Production Area B",
    "Utility Section",
    "Drilling Site",
    "Loading Terminal"
]

locations = [
    "compressor bay",
    "maintenance area",
    "pipeline corridor",
    "tank farm",
    "loading point",
    "workshop",
    "production unit",
    "electrical room",
    "elevated work platform",
    "process line",
    "pump area",
    "utility yard",
    "valve station",
    "storage area",
    "equipment bay"
]

activities = [
    "compressor maintenance",
    "pipeline repair",
    "electrical maintenance",
    "crane operation",
    "lifting operation",
    "tank entry",
    "vessel cleaning",
    "grinding",
    "welding",
    "vehicle movement"
]

workers = [
    "a maintenance technician",
    "the electrical crew",
    "a contractor",
    "the lifting team",
    "a field operator",
    "the inspection team",
    "a welding crew",
    "the maintenance group",
    "a vehicle operator",
    "the site team"
]

times = [
    "during the morning shift",
    "during the afternoon shift",
    "before the evening shutdown",
    "during a planned maintenance window",
    "at the start of the work period",
    "shortly after the crew arrived",
    "during routine field activity",
    "while preparing the work area"
]

# =========================================================
# SAFE / NO-SIF COMPONENTS
# =========================================================

no_actions = [
    "reviewed the work instructions before starting",
    "checked the condition of the equipment before use",
    "confirmed the work boundary with the supervisor",
    "verified the required authorization",
    "inspected the surrounding area before beginning",
    "confirmed communication with the control room",
    "checked the condition of the lifting equipment",
    "reviewed the isolation status before starting",
    "confirmed that the access route was clear",
    "completed the pre-job safety discussion"
]

no_controls = [
    "the required safeguards were available",
    "the work area remained controlled",
    "the planned precautions were maintained",
    "the equipment was within the approved operating condition",
    "the required separation was maintained",
    "the access controls remained in place",
    "the team stayed within the designated work area",
    "the approved sequence was followed",
    "the supervisor remained available",
    "the work authorization was valid"
]

no_observations = [
    "No hazardous deviation was observed.",
    "The activity continued under the planned controls.",
    "The observation did not identify a credible high-consequence exposure.",
    "The work proceeded without an uncontrolled change in conditions.",
    "No significant control breakdown was identified.",
    "The crew completed the activity without a serious exposure.",
    "The surrounding conditions remained stable throughout the task.",
    "The planned barriers remained effective during the observation."
]

# =========================================================
# HIGH-RISK / YES COMPONENTS
# =========================================================

yes_actions = [
    "started the task before the required control had been confirmed",
    "moved into the work area while an important safeguard was still being checked",
    "continued the activity after a required barrier was found to be ineffective",
    "positioned personnel closer to the active equipment than permitted",
    "began work before the isolation status was fully verified",
    "approached the operating equipment without maintaining the expected separation",
    "continued preparation while a critical control remained incomplete",
    "entered the restricted area before the required authorization was confirmed",
    "started the operation while another crew was still working inside the affected area",
    "proceeded with the task after an unexpected change in conditions"
]

yes_hazards = [
    "unexpected movement of equipment",
    "possible release of stored energy",
    "uncontrolled movement of a suspended load",
    "exposure to an energized component",
    "loss of separation from operating machinery",
    "unexpected pressure in the process line",
    "movement of a vehicle through the work area",
    "temporary loss of a protective barrier",
    "an unplanned change in the work environment",
    "an active energy source that had not been fully controlled"
]

yes_interventions = [
    "The supervisor stopped the activity and moved the crew to a safe location.",
    "Work was suspended until the missing control was restored.",
    "The team withdrew from the area and reassessed the work conditions.",
    "The activity was paused and the affected equipment was secured.",
    "The supervisor intervened before the task could continue.",
    "The crew stopped work and the control measures were reviewed.",
    "The area was isolated while the condition was investigated.",
    "The operation was halted until the required protection was established."
]

# =========================================================
# HARD NEGATIVE COMPONENTS
# These look unsafe but do NOT indicate SIF potential.
# =========================================================

hard_negative_events = [
    "A worker noticed a loose hand tool on the work platform and secured it before the task began.",
    "A small housekeeping issue was observed beside the work area and was corrected immediately.",
    "The crew found minor surface damage on a non-critical component during inspection and reported it for maintenance.",
    "A temporary access obstruction was identified and removed before personnel entered the area.",
    "A warning sign had shifted from its original position and was restored before work started.",
    "A worker reported a minor equipment defect that did not affect the active task and the item was tagged for follow-up.",
    "A routine inspection identified a small housekeeping deviation with no personnel exposure.",
    "The team noticed a damaged label on equipment and replaced it before continuing."
]

hard_negative_controls = [
    "No person was exposed to the condition.",
    "The affected item was outside the active work path.",
    "The task had not started when the issue was identified.",
    "The equipment remained isolated from personnel activity.",
    "The condition was corrected before it could affect the work.",
    "The observation involved housekeeping rather than an active energy source.",
    "The issue had no impact on the work boundary.",
    "The crew maintained the required separation throughout."
]

# =========================================================
# LIFE SAVING RULES
# =========================================================

rules = [
    "Energy Isolation",
    "Line of Fire",
    "Hot Work",
    "Confined Space"
]

# =========================================================
# GENERATORS
# =========================================================

def create_no_report(i):

    site = random.choice(sites)
    location = random.choice(locations)
    activity = random.choice(activities)
    worker = random.choice(workers)
    time = random.choice(times)

    # 35% hard negatives
    if random.random() < 0.35:

        event = random.choice(hard_negative_events)
        control = random.choice(hard_negative_controls)

        text = (
            f"{time}, at the {site}, {event} "
            f"The observation was associated with {activity} near the {location}. "
            f"{control} "
            f"The team corrected the issue before normal work continued."
        )

    else:

        action = random.choice(no_actions)
        control = random.choice(no_controls)
        observation = random.choice(no_observations)

        text = (
            f"{time}, at the {site}, {worker} was involved in {activity} "
            f"near the {location}. "
            f"Before the task progressed, the team {action}. "
            f"{control.capitalize()}. "
            f"{observation}"
        )

    add_report(
        i=i,
        report_type=random.choice([
            "Safety Observation",
            "Unsafe Condition",
            "Unsafe Act",
            "Near Miss"
        ]),
        site=site,
        activity=activity,
        text=text,
        label="NO",
        precursor="No credible precursor identified",
        evidence=random.choice([
            "Control verified",
            "Procedure followed",
            "Precaution confirmed",
            "Work area controlled",
            "No personnel exposure"
        ]),
        rule=random.choice(rules),
        priority=random.choice([
            "Routine",
            "Low"
        ]),
        review="No"
    )


def create_yes_report(i):

    site = random.choice(sites)
    location = random.choice(locations)
    activity = random.choice(activities)
    worker = random.choice(workers)
    time = random.choice(times)

    action = random.choice(yes_actions)
    hazard = random.choice(yes_hazards)
    intervention = random.choice(yes_interventions)

    text = (
        f"{time}, at the {site}, {worker} was involved in {activity} "
        f"near the {location}. "
        f"The worker or crew {action}. "
        f"This created a situation involving {hazard}. "
        f"{intervention}"
    )

    add_report(
        i=i,
        report_type=random.choice([
            "Unsafe Condition",
            "Unsafe Act",
            "Near Miss"
        ]),
        site=site,
        activity=activity,
        text=text,
        label="YES",
        precursor=random.choice([
            "Barrier failure",
            "Control failure",
            "Energy exposure",
            "Unsafe positioning",
            "Uncontrolled hazard"
        ]),
        evidence=random.choice([
            "Control not verified",
            "Barrier ineffective",
            "Personnel exposed to hazard",
            "Unsafe position observed",
            "Unexpected hazard identified"
        ]),
        rule=random.choice(rules),
        priority=random.choice([
            "High",
            "Critical"
        ]),
        review="Yes"
    )


def add_report(i, report_type, site, activity, text,
               label, precursor, evidence,
               rule, priority, review):

    reports.append({
        "report_id": f"V3-{i:04d}",
        "report_type": report_type,
        "site": site,
        "activity": activity,
        "report_text": text,
        "language": "English",
        "sif_potential": label,
        "precursor": precursor,
        "evidence": evidence,
        "life_saving_rule": rule,
        "priority": priority,
        "human_review": review
    })


# =========================================================
# GENERATE 1500 REPORTS
# =========================================================

for i in range(1, 751):
    create_no_report(i)

for i in range(751, 1501):
    create_yes_report(i)


# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(reports)

df = df.sample(
    frac=1,
    random_state=2026
).reset_index(drop=True)


# =========================================================
# DUPLICATE CHECK
# =========================================================

duplicate_count = df["report_text"].duplicated().sum()
unique_count = df["report_text"].nunique()


print("\n========================================")
print("SIF-SANKET V3 DATASET")
print("========================================")

print("\nTotal reports:", len(df))

print("\nLabel distribution:")
print(df["sif_potential"].value_counts())

print("\nExact duplicate report texts:", duplicate_count)

print("Unique report texts:", unique_count)

print("\nYES examples:")
print(
    df[df["sif_potential"] == "YES"]
    ["report_text"]
    .head(3)
    .to_string(index=False)
)

print("\nNO examples:")
print(
    df[df["sif_potential"] == "NO"]
    ["report_text"]
    .head(3)
    .to_string(index=False)
)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("Saved successfully:")
print(OUTPUT_FILE)
print("========================================")
