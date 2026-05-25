# Topic ID: ENV-03: Diurnal Ozone Cycles

# Research Title: Automated Statistical Analysis of Diurnal Ozone Cycle Patterns in Urban Atmospheres Using Python Data Pipelines

# Technological University of the Philippines – Manila | Electronics Engineering Department

# Project Overview 

This project presents an automated Python-based data pipeline for analyzing ground-level ozone (O₃) concentration patterns in urban environments. The system processes hourly air quality telemetry data, performs statistical analysis using NumPy, and generates visualizations to characterize the diurnal ozone cycle in Los Angeles, United States.

# The pipeline automates:
- Data ingestion
- Data cleaning and preprocessing
- Time-period classification
- Statistical analysis
- Outlier detection
- Correlation analysis
- Static and animated visualizations

The study focuses on ozone telemetry records collected from November 4–18, 2025.

# Features

- Full diurnal ozone cycle analysis for Los Angeles
- NumPy-powered statistical analysis (skewness, outliers, correlations)
- 4 publication-ready static visualizations
- 2 professional GIF animations
- OOP pipeline with method chaining
- Automated data cleaning and preprocessing
- Sample telemetry table export
- WHO guideline visualization

# How to Run

1. Clone the Repository
- git clone https://github.com/manzanoseankaylee-tech/EDS_4761_MANZANO.git
- cd EDS_4761_MANZANO

2. Install Required Libraries
- pip install -r requirements.txt

3. Download the Dataset
Go to Kaggle – Diurnal Ozone Cycles Data and download:

- globalAirQuality.csv
- Place files inside the data/ folder.

4. Update the Project Root Path
Open main.py and update the project_root variable to match your local directory:

project_root = Path("your/local/path/EDS_4761_MANZANO")

5. Run the Pipeline
python main.py
All outputs (cleaned dataset, static plots, and animated GIFs) will be automatically saved to the outputs/ folder.

# Project Structure
EDS_4761_Manzano/
├── main.py                    
├── data/
│   ├── globalAirQuality.csv   # Raw data
│   └── dataset_cleaned.csv    # Processed data
├── outputs/
│   ├── Figure1_Diurnal_Boxplot.png
│   ├── Figure2_Hourly_Avg_Line.png
│   ├── Figure3_Correlation_Heatmap.png
│   ├── Figure4_O3_vs_NO2_Scatter.png
│   ├── Animation1_Diurnal_Evolution.gif
│   ├── Animation2_Heatmap_Series.gif
│   └── Table1_Sample_Records.csv
├── requirements.txt
└── .gitignore



# AI Disclosure

Artificial Intelligence (AI) Usage Disclosure
This project utilized Artificial Intelligence (AI) tools as supplementary assistants during the development, debugging, documentation, formatting, and improvement stages of the study. The following AI systems were used throughout the project:

Grok - https://grok.com/
ChatGPT — https://chatgpt.com
Claude — https://claude.ai
Microsoft CoPilot - https://copilot.microsoft.com/
These tools were primarily used for:

- Code debugging and syntax assistance
- Documentation refinement and proofreading
- Suggestions for visualization formatting and project structure

All final decisions, implementations, data processing, analysis, interpretation of results, and conclusions in this project were independently reviewed, validated, and finalized by the author. AI-generated suggestions were used only as assistive references and were not considered substitutes for critical analysis, programming logic, or academic judgment.

The author remains fully responsible for the accuracy, originality, integrity, and overall content of this project.

# Author
Sean Kaylee Manzano | TUPM-25-4761
Electronics Engineering Department
Technological University of the Philippines, Manila
seankaylee.manzano@tup.edu.ph
