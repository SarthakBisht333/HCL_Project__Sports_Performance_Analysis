""" Project: SPORTS PERFORMANCE ANALYSIS USING Python,NumPy and Pandas
1. Aim
To design and implement a Python-based data analytics system that generates structured multi-sport datasets, evaluates athlete and team performance across five sports using Pandas and NumPy, and visualizes key metrics through custom terminal-based bar charts.
2. Tools & Libraries Used
Programming Language: Python 3
Pandas: Used for creating CSV datasets, tabular data manipulation, groupby() aggregations, sorting, and conditional filtering.
NumPy: Used for vectorized numerical computations, rounding, statistical analysis (Mean, Median, Standard Deviation), and Pearson Correlation.
3. Dataset Description
The program programmatically generates two relational CSV files covering five sports (Cricket, Football, Basketball, Badminton, and Athletics):

athletes_data.csv: Contains 10 records with attributes Athlete_ID, Name, Sport, Key_Stat, Matches, Wins, Perf_Score, Stamina, and Training_Hrs.

tournament_matches.csv: Contains 10 match records with attributes Match_ID, Sport, Team_A_Pts, Team_B_Pts, and Audience.
4. Methodology

Automated Data Generation: Dictionaries containing multi-sport records are converted into DataFrames and exported to .csv files using to_csv().

Derived Metrics (Feature Engineering): Two composite metrics are computed using NumPy operations:

Efficiency Index = (0.4 * Perf_Score)+(0.4 * Win_Rate)+(0.2 * Stamina)
Analytical Queries (7 Insights): Evaluates sport-wise averages, top-3 overall athletes, score consistency (Mean, Median, Std Dev), impact of weekly training hours on performance (Correlation), elite athlete filtering, audience engagement, and top-performing sport.

Custom Data Visualization (5 Graphs): Implemented a lightweight plotting function using Unicode block characters (█) scaled proportionally to maximum values to display five comparative bar graphs directly in the console.
Conclusion: The project demonstrates end-to-end exploratory data analysis without relying on heavy external visualization or machine learning libraries. The analysis confirms a strong positive correlation between weekly training hours and athlete performance scores, while the weighted Efficiency Index provides an objective framework to compare athletes across different sports.
"""



import numpy as np
import pandas as pd

ATHLETES_CSV = "athletes_data.csv"
MATCHES_CSV = "tournament_matches.csv"


# 1. Auto-generate 2 Multi-Sport CSV Datasets
def create_datasets():
    athletes = {
        "Athlete_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Name": ["Aarav", "Rohan", "Kabir", "Vihaan", "Arjun", "Dev", "Ishaan", "Kunal", "Aditya", "Nikhil"],
        "Sport": [
            "Cricket", "Cricket",
            "Football", "Football",
            "Basketball", "Basketball",
            "Badminton", "Badminton",
            "Athletics", "Athletics"
        ],
        "Key_Stat": [
            "265 Runs", "190 Runs",
            "7 Goals", "5 Goals",
            "118 Pts", "96 Pts",
            "14 Sets", "11 Sets",
            "10.7s 100m", "11.1s 100m"
        ],
        "Matches": [8, 8, 8, 8, 8, 8, 8, 8, 6, 6],
        "Wins": [6, 4, 5, 4, 7, 5, 6, 4, 5, 3],
        "Perf_Score": [88, 76, 84, 75, 92, 81, 86, 74, 90, 78],
        "Stamina": [85, 78, 91, 84, 89, 82, 87, 80, 95, 88],
        "Training_Hrs": [14, 10, 16, 12, 18, 13, 15, 11, 17, 12]
    }

    matches = {
        "Match_ID": ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10"],
        "Sport": [
            "Cricket", "Cricket",
            "Football", "Football",
            "Basketball", "Basketball",
            "Badminton", "Badminton",
            "Athletics", "Athletics"
        ],
        "Team_A_Pts": [182, 165, 3, 2, 78, 64, 21, 19, 95, 88],
        "Team_B_Pts": [174, 170, 2, 2, 72, 69, 17, 21, 91, 84],
        "Audience": [450, 420, 380, 350, 310, 290, 220, 240, 340, 310]
    }

    pd.DataFrame(athletes).to_csv(ATHLETES_CSV, index=False)
    pd.DataFrame(matches).to_csv(MATCHES_CSV, index=False)
    print(f"[INFO] 2 CSV files created: '{ATHLETES_CSV}' & '{MATCHES_CSV}'\n")


# Helper function to render clean horizontal bar charts in terminal
def plot_terminal_graph(title, labels, values, unit=""):
    print(f"\n=== {title} ===")
    max_val = max(values) if max(values) > 0 else 1
    for label, val in zip(labels, values):
        bar_len = int((val / max_val) * 30)
        bar = "█" * bar_len
        print(f"{str(label):<13} | {bar:<30} {val}{unit}")


def analyze_sports():
    df_athletes = pd.read_csv(ATHLETES_CSV)
    df_matches = pd.read_csv(MATCHES_CSV)

    # Derived metrics using NumPy
    df_athletes["Win_Rate"] = np.round((df_athletes["Wins"] / df_athletes["Matches"]) * 100, 1)
    df_athletes["Efficiency_Index"] = np.round(
        (0.4 * df_athletes["Perf_Score"]) + (0.4 * df_athletes["Win_Rate"]) + (0.2 * df_athletes["Stamina"]), 2
    )

    print("=================================================")
    print("     MULTI-SPORT PERFORMANCE ANALYSIS REPORT")
    print("=================================================")

    # Q1: Sport-wise Average Performance, Stamina & Win Rate
    sport_summary = (
        df_athletes.groupby("Sport")[["Perf_Score", "Stamina", "Win_Rate"]]
        .mean()
        .round(2)
        .reset_index()
    )
    print("\nQ1. Sport-Wise Average Performance, Stamina & Win Rate:")
    print(sport_summary.to_string(index=False))

    # Q2: Top 3 Overall Athletes by Efficiency Index
    top_athletes = df_athletes.sort_values(by="Efficiency_Index", ascending=False).head(3)
    print("\nQ2. Top 3 Overall Athletes (Efficiency Index Ranking):")
    print(top_athletes[["Name", "Sport", "Key_Stat", "Win_Rate", "Efficiency_Index"]].to_string(index=False))

    # Q3: Statistical Consistency of Performance Scores (NumPy)
    mean_perf = np.round(np.mean(df_athletes["Perf_Score"]), 2)
    median_perf = np.round(np.median(df_athletes["Perf_Score"]), 2)
    std_perf = np.round(np.std(df_athletes["Perf_Score"]), 2)
    print(f"\nQ3. Overall Performance Score Stats (NumPy):")
    print(f"    Mean: {mean_perf} | Median: {median_perf} | Std Deviation: {std_perf}")

    # Q4: Correlation between Weekly Training Hours & Performance Score
    corr = np.round(np.corrcoef(df_athletes["Training_Hrs"], df_athletes["Perf_Score"])[0, 1], 2)
    print(f"\nQ4. Training Hours vs Performance Score Correlation: {corr} (Strong Positive Impact)")

    # Q5: High-Stamina & High-Win Athletes Filtering
    elite = df_athletes[(df_athletes["Stamina"] >= 85) & (df_athletes["Win_Rate"] >= 60)]
    print("\nQ5. Elite Athletes (Stamina >= 85 & Win Rate >= 60%):")
    print(elite[["Name", "Sport", "Stamina", "Win_Rate"]].to_string(index=False))

    # Q6: Sport-wise Audience Engagement & Closest Contests
    df_matches["Margin"] = np.abs(df_matches["Team_A_Pts"] - df_matches["Team_B_Pts"])
    match_summary = (
        df_matches.groupby("Sport")
        .agg(Avg_Audience=("Audience", "mean"), Avg_Margin=("Margin", "mean"))
        .round(1)
        .reset_index()
    )
    print("\nQ6. Sport-Wise Audience Turnout & Match Closeness:")
    print(match_summary.to_string(index=False))

    # Q7: Best Performing Sport Overall
    best_sport = sport_summary.loc[sport_summary["Perf_Score"].idxmax()]
    print(
        f"\nQ7. Top Performing Sport: {best_sport['Sport']} "
        f"(Avg Perf Score: {best_sport['Perf_Score']}, Avg Win Rate: {best_sport['Win_Rate']}%)"
    )

    # --- 5 VISUAL GRAPHS (Terminal Bar Charts) ---
    print("\n=================================================")
    print("           VISUAL PERFORMANCE GRAPHS")
    print("=================================================")

    plot_terminal_graph("Graph 1: Sport-Wise Average Performance Score", sport_summary["Sport"], sport_summary["Perf_Score"], " pts")

    top_5 = df_athletes.sort_values(by="Efficiency_Index", ascending=False).head(5)
    plot_terminal_graph("Graph 2: Top 5 Athletes by Efficiency Index", top_5["Name"] + " (" + top_5["Sport"].str[:3] + ")", top_5["Efficiency_Index"], " idx")

    plot_terminal_graph("Graph 3: Sport-Wise Average Stamina Score", sport_summary["Sport"], sport_summary["Stamina"], "/100")

    plot_terminal_graph("Graph 4: Sport-Wise Average Audience Attendance", match_summary["Sport"], match_summary["Avg_Audience"], " fans")

    plot_terminal_graph("Graph 5: Weekly Training Hours per Athlete", df_athletes["Name"], df_athletes["Training_Hrs"], " hrs")


if __name__ == "__main__":
    create_datasets()
    analyze_sports()



output:
<img width="341" height="366" alt="Screenshot 2026-11-20 221147" src="https://github.com/user-attachments/assets/25f2c28a-2534-4ed5-b4ca-341050fd7e3b" />
<img width="244" height="386" alt="Screenshot 2026-11-20 221236" src="https://github.com/user-attachments/assets/c9b6c07c-e490-42eb-b65c-0711c37e5027" />
<img width="575" height="419" alt="Screenshot 2026-11-20 222401" src="https://github.com/user-attachments/assets/38484033-26f9-4b6c-b710-06db5ff1f495" />
<img width="506" height="416" alt="Screenshot 2026-11-20 222424" src="https://github.com/user-attachments/assets/50bbde08-e76b-4639-999b-07845d1f1d26" />
<img width="581" height="389" alt="Screenshot 2026-11-20 222442" src="https://github.com/user-attachments/assets/055836b7-dc1d-4a2d-95fc-56b8b1d9d336" />
<img width="539" height="401" alt="Screenshot 2026-11-20 222502" src="https://github.com/user-attachments/assets/92a35ac0-0b89-49e5-8edd-669f47fe4694" />
<img width="459" height="402" alt="Screenshot 2026-11-20 222518" src="https://github.com/user-attachments/assets/4c726e96-5537-4694-b927-367c92fc3cb0" />
<img width="305" height="164" alt="Screenshot 2026-11-20 222531" src="https://github.com/user-attachments/assets/c2dc9756-d326-4823-832b-92a89c9406d1" />


# Name- Sarthak Bisht
# Admission Number- 2026b15410201
# Branch and Section- CSE(DS)--3
