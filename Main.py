# Import necessary libraries
import pandas as pd
import pdfplumber as pdf_plum
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
pdf_path = 'LatestListofScheduledtribes.pdf'
csv_path = 'dataset.csv'

# Global variables for tracking statistics
state_match_counts = defaultdict(int)  # Track matches found for each state
state_total_counts = defaultdict(int)  # Track total records processed for each state
caste_frequency = defaultdict(int)  # Track how often each caste appears
state_unique_castes = defaultdict(set)  # Track unique castes per state


# PDF Processing Class
class LoadPDF:
    """
    A class to handle extraction and parsing of data from PDF files.
    """

    @staticmethod
    def extract_text(file_path):
        """Extract all text from a given PDF file."""
        with pdf_plum.open(file_path) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages)
        return text

    @staticmethod
    def parse_states_and_castes(text):
        """Parse states and their respective castes from the extracted PDF text."""
        state_names = ['MAHARASHTRA', 'Rajasthan', 'UttarPradesh']  # List of states to match
        state_data = re.split('|'.join(state_names), text)  # Split text by states
        state_data = [data.strip() for data in state_data if data.strip()]  # Clean data

        state_caste_dict = {}

        # Iterate over states and match castes
        for idx, state in enumerate(state_names):
            if idx < len(state_data):
                # Extract castes using regex pattern
                caste_names = re.findall(r'\d+\.\s*([a-zA-Z,\s]+)', state_data[idx])
                cleaned_castes = [name.strip().upper() for match in caste_names for name in re.split(r'[,\n]', match) if
                                  name.strip().upper()]

                # Store the cleaned caste names for the state
                state_caste_dict[state.lower()] = cleaned_castes

                # Count caste frequencies and track unique castes per state
                for caste in cleaned_castes:
                    caste_frequency[caste] += 1
                    state_unique_castes[state.lower()].add(caste)

        return state_caste_dict

    @staticmethod
    def get_state_with_caste():
        """Return a dictionary of states and their respective castes from the PDF."""
        return LoadPDF.parse_states_and_castes(LoadPDF.extract_text(pdf_path))


# Load and validate the CSV data
df = pd.read_csv(csv_path)
required_cols = ['Full Name', 'Father Name', 'Mother Name', 'Spouse Name', 'State']

# Check if the CSV has the required columns
if not all(col in df.columns for col in required_cols):
    raise ValueError(
        "The CSV file must contain columns for Full Name, Father Name, Mother Name, Spouse Name, and State."
    )

# Extract caste data from the PDF
pdf_data_dict = LoadPDF.get_state_with_caste()

# Add a new column to indicate whether a Scheduled Tribe (ST) name match is found
df['ST_Name_Match'] = False


# Helper functions
def contains_match(name_parts, caste_list):
    """Check if any part of the name matches with any caste in the caste list."""
    for name in name_parts:
        name_parts_split = name.split()
        for part in name_parts_split:
            if any(part.lower() == caste_part.lower() for caste in caste_list for caste_part in caste.split()):
                return True
    return False


def extract_name_parts(full_name):
    """Extract parts of a full name."""
    if isinstance(full_name, str) and full_name.strip():
        return full_name.strip().split()
    return []


# Process each row in the DataFrame
for index, row in df.iterrows():
    state = row['State'].strip()
    state_total_counts[state] += 1  # Track total rows processed for the state

    # Check if the state exists in the PDF data
    if state.lower() in pdf_data_dict:
        st_castes = pdf_data_dict[state.lower()]
        names_to_check = [row['Full Name'], row['Father Name'], row['Mother Name'], row['Spouse Name']]

        # Check if any part of the names matches a caste in the caste list
        match_found = any(contains_match(extract_name_parts(name), st_castes) for name in names_to_check if
                          isinstance(name, str) and name.strip())
        df.at[index, 'ST_Name_Match'] = match_found
        if match_found:
            state_match_counts[state] += 1  # Count successful matches

# State-wise Analysis
print("\nState-wise Analysis:")
states = []
match_percentages = []
for state in state_total_counts:
    total = state_total_counts[state]
    matches = state_match_counts[state]
    percentage = (matches / total) * 100 if total > 0 else 0
    states.append(state)
    match_percentages.append(percentage)
    print(f"{state}: {matches}/{total} matches ({percentage:.2f}%)")

# Caste-wise Analysis
print("\nCaste-wise Analysis:")
most_frequent_caste = max(caste_frequency, key=caste_frequency.get)
least_frequent_caste = min(caste_frequency, key=caste_frequency.get)
print(f"Most frequently appearing caste: {most_frequent_caste} (appears {caste_frequency[most_frequent_caste]} times)")
print(
    f"Least frequently appearing caste: {least_frequent_caste} (appears {caste_frequency[least_frequent_caste]} times)")

# Unique castes per state
print("\nUnique castes in each state:")
unique_caste_counts = []
for state, castes in state_unique_castes.items():
    count = len(castes)
    unique_caste_counts.append(count)
    print(f"{state.capitalize()}: {count} unique castes")

# Customer IDs marked as ST according to the PDF
st_matches = df[(df['ST_Name_Match'] == True) & (df['Caste'] == 'General')]['Customer ID'].nunique()
print(f"\nNumber of distinct Customer IDs marked as ST according to the PDF: {st_matches}")

# State-wise count of distinct Loan IDs for ST matches
state_loan_counts = df[df['ST_Name_Match'] == True].groupby('State')['Loan ID'].nunique().reset_index()
state_loan_counts.columns = ['State', 'ST Loan Count']
state_loan_counts = state_loan_counts.sort_values('ST Loan Count', ascending=False)
print("\nState-wise count of distinct Loan IDs where any customer ID is marked as ST as per PDF:")
print(state_loan_counts.to_string(index=False))

# Save state-wise loan counts to CSV
state_loan_counts.to_csv('state_wise_st_loan_counts.csv', index=False)
print("\nState-wise table saved to 'state_wise_st_loan_counts.csv'")

# Summary statistics
total_st_loans = state_loan_counts['ST Loan Count'].sum()
max_st_loans = state_loan_counts['ST Loan Count'].max()
min_st_loans = state_loan_counts['ST Loan Count'].min()
avg_st_loans = state_loan_counts['ST Loan Count'].mean()
print(f"\nSummary Statistics:")
print(f"Total ST Loans across all states: {total_st_loans}")
print(f"State with most ST Loans: {state_loan_counts.iloc[0]['State']} ({max_st_loans} loans)")
print(f"State with least ST Loans: {state_loan_counts.iloc[-1]['State']} ({min_st_loans} loans)")
print(f"Average ST Loans per state: {avg_st_loans:.2f}")

# Visualizations

# Bar plot for state-wise match percentages
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.bar(states, match_percentages)
plt.title('State-wise Match Percentages')
plt.xlabel('States')
plt.ylabel('Match Percentage')
plt.xticks(rotation=45)

# Bar plot for top 10 caste frequencies
plt.subplot(2, 2, 2)
caste_names = list(caste_frequency.keys())
caste_counts = list(caste_frequency.values())
plt.bar(caste_names[:10], caste_counts[:10])
plt.title('Top 10 Caste Frequencies')
plt.xlabel('Castes')
plt.ylabel('Frequency')
plt.xticks(rotation=90)

# Caste Frequency Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(caste_counts[:10], labels=caste_names[:10], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Caste Frequency Distribution (Top 10 Castes)')
plt.tight_layout()
plt.show()

# Caste Frequency Histogram
plt.figure(figsize=(10, 6))
plt.hist(caste_counts, bins=20)
plt.title('Histogram of Caste Appearance Frequency')
plt.xlabel('Frequency of Appearance')
plt.ylabel('Number of Castes')
plt.tight_layout()
plt.show()

# Heatmap for State-wise Caste Frequencies
state_caste_df = pd.DataFrame({state: [len(castes)] for state, castes in state_unique_castes.items()})
plt.figure(figsize=(12, 8))
sns.heatmap(state_caste_df, annot=True, cmap='coolwarm', fmt='d')
plt.title('Heatmap of Unique Caste Counts per State')
plt.show()
