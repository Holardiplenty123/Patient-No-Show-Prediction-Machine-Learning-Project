# 🏥 Patient No-Show Prediction — Machine Learning Project

Predicting the probability that a patient will miss a scheduled medical appointment, using demographic, health, and scheduling data.

---

## 📌 Problem Statement

Hospitals and clinics face a significant operational and financial problem when patients book appointments but do not show up. No-shows lead to:

- Idle doctors and staff
- Wasted appointment slots that could have gone to other patients
- Longer waiting lists
- Direct revenue loss

**Goal:** Build a machine learning model that estimates the probability of a patient not showing up, so hospitals can proactively intervene (e.g. reminder calls, overbooking strategies, targeted SMS) to reduce wasted resources, improve efficiency, and protect revenue.

---

## 📊 Dataset

- **Source:** [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Kaggle)
- **Size:** 110,527 rows × 14 columns
- **Target variable:** `No-show` (Yes / No)
- **Class balance:** ~80% show, ~20% no-show (imbalanced)

**Key raw features:**

| Feature | Description |
|---|---|
| `Gender` | Patient gender |
| `Age` | Patient age |
| `Scholarship` | Enrolled in Bolsa Família welfare program |
| `Hipertension` | Hypertension diagnosis |
| `Diabetes` | Diabetes diagnosis |
| `Alcoholism` | Alcoholism history |
| `Handcap` | Disability indicator |
| `SMS_received` | Whether an SMS reminder was sent |
| `ScheduledDay` / `AppointmentDay` | Used to engineer waiting time |

---

## 🧹 Data Preparation

- Standardized column names (lowercase, underscores)
- Removed invalid rows (negative ages)
- Dropped identifier columns (`patientid`, `appointmentid`) not useful for modeling
- Verified there were no missing values across the dataset

## 🛠️ Feature Engineering

- Converted `scheduledday` and `appointmentday` to datetime
- Engineered **`waiting_days`** = days between booking and appointment date (this turned out to be the single most predictive feature)
- Encoded target (`no_show`) and `gender` as binary
- Final feature set: `gender`, `age`, `scholarship`, `hipertension`, `diabetes`, `alcoholism`, `handcap`, `sms_received`, `waiting_days`

## 🔍 Exploratory Data Analysis

- Class distribution (show vs. no-show)
- Age distribution across the patient population
- Age and waiting-time patterns by no-show status
- No-show rate by gender

---

## 🤖 Modeling

Three baseline classifiers were trained on an 80/20 stratified train-test split (features scaled where required):

- Logistic Regression
- Random Forest
- K-Nearest Neighbors

Because the dataset is imbalanced (~20% no-shows), models were then retrained with `class_weight='balanced'` (Logistic Regression, Random Forest) to improve minority-class recall.

### Baseline Results

| Model | Accuracy | Precision (No-show) | Recall (No-show) | F1 (No-show) |
|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.36 | 0.01 | 0.03 |
| Random Forest | 0.77 | 0.32 | 0.17 | 0.22 |
| K-Nearest Neighbors | 0.75 | 0.29 | 0.16 | 0.20 |

The baseline models are heavily biased toward the majority class — high overall accuracy but almost no ability to actually catch no-shows (the class that matters most operationally).

### Balanced Results (class-weighted)

| Model | Accuracy | Precision (No-show) | Recall (No-show) | F1 (No-show) |
|---|---|---|---|---|
| Logistic Regression (Balanced) | 0.67 | 0.32 | 0.57 | 0.41 |
| Random Forest (Balanced) | 0.75 | 0.31 | 0.23 | 0.26 |

Balancing class weights trades some overall accuracy for a large jump in no-show recall (Logistic Regression: 1% → 57%) — a more useful trade-off for a hospital that wants to flag as many likely no-shows as possible for intervention.

---

## ⭐ Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| `waiting_days` | 0.544 |
| `age` | 0.378 |
| `sms_received` | 0.023 |
| `gender` | 0.016 |
| `hipertension` | 0.009 |
| `handcap` | 0.009 |
| `diabetes` | 0.008 |
| `scholarship` | 0.007 |
| `alcoholism` | 0.006 |

**Key insight:** The number of days between booking and the actual appointment (`waiting_days`) is by far the strongest predictor of a no-show, followed by patient age. Demographic and health-condition flags contribute comparatively little on their own.

---

## 🧰 Tech Stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (Logistic Regression, Random Forest, KNN)
- **Environment:** Jupyter / Google Colab

---

## 📁 Project Structure
