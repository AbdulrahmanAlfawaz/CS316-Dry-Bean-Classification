# Dry Bean Classification for Sustainable Sorting
CS316 Introduction to AI and Data Science, Term 261
Abdulrahman Alfawaz, individual project

## Purpose
Predict seven bean varieties from 16 numeric measurements. This is a small educational prototype for sorting support. Its potential sustainability contribution is more efficient handling under SDG 12. Actual reductions in food loss, energy or cost have not been measured.

## Run locally
Install Python 3.12. Extract the complete folder, open a terminal in it and run:

```bash
python -m pip install -r requirements.txt
python project.py
```

The dataset is included, so execution does not download data. Results are saved in results/. The main script resolves paths relative to its own location. The supplied notebook can also run the script when placed beside it. A complete run usually takes under a few minutes, depending on the computer.

## Google Colab
Upload the project ZIP to Colab and unzip it. In a cell, change directory into CS316_Bean_Project using `%cd CS316_Bean_Project`, install requirements using `!pip install -r requirements.txt`, then run `!python project.py`.

## Method
Remove 68 exact duplicates, retain 13,543 records, split 80/20 with stratification and seed 42. Compare logistic regression, KNN and decision tree using five-fold training cross-validation. Select by macro F1. Fit preprocessing inside each fold. Evaluate the winner once on the test set.

## Measured results
Logistic regression C=10 wins by a very small cross-validation margin. Test accuracy is 92.14%, macro F1 is 0.9316, and the majority baseline is 26.17%. The small CV difference from KNN does not demonstrate statistical superiority. All outputs come from the included script.

## Files
- project.py: commented, executable analysis
- project.ipynb: notebook entry point and explanations
- data/: included data and attribution
- results/: metrics, predictions and charts
- Report.pdf and Report.tex: IEEE report and editable source
- Presentation.pptx: 10 slides with speaker notes
- Proposal.pdf: proposal covering phase 1
- Explanation_Guide.pdf: run instructions, decisions and questions
- SUBMISSION_CHECKLIST.md: remaining submission actions

## GitHub
Create a repository, upload the contents of this folder, and share the repository URL with the instructor. Keep README.md at the repository root. The repository has not yet been published from this environment.

## AI use
ChatGPT assisted with topic selection, code generation, execution, analysis, drafting and presentation preparation. The student must review and understand the work and disclose this assistance according to course rules. No plagiarism or AI-detector score has been obtained.
