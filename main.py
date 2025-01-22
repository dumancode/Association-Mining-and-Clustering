from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import DBSCAN



data = []
with open("Apple_sequence_dataset.txt", "r") as file:
    for line in file:
        segments = [segment.strip().split(",")[0] for segment in line.strip()[2:-2].split("], [")]
        data.append(segments)

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.75, use_colnames=True)

max_length = frequent_itemsets['itemsets'].apply(lambda x: len(x)) == max(frequent_itemsets['itemsets'].apply(len))

rules = association_rules(frequent_itemsets[max_length], metric="confidence", min_threshold=0.81, support_only=True)

print("\nAssociation Rules with Confidence:")
print(rules[['antecedents', 'consequents']])

data_dir = 'Apple_fixation_dataset'
data_files = [f for f in os.listdir(data_dir) if f.startswith('P-') and f.endswith('.txt')]

all_data = []
for file in data_files:
    file_path = os.path.join(data_dir, file)
    # Check if the file is not empty
    if os.path.getsize(file_path) > 0:
        try:

            df = pd.read_csv(file_path, delimiter='\t')
            required_columns = ['FixationIndex', 'Timestamp', 'FixationDuration', 'MappedFixationPointX', 'MappedFixationPointY', 'StimuliName']
            if not df.empty and all(column in df.columns for column in required_columns):
                df['Participant'] = file  # Add a column for participant identifier
                all_data.append(df)

            else:
                print(f"Warning: {file} does not contain the required columns or is empty and will be skipped.")
        except pd.errors.ParserError:
            print(f"Warning: {file} is not in the expected format and will be skipped.")
        except pd.errors.EmptyDataError:
            print(f"Warning: {file} is empty or not in the expected format and will be skipped.")
    else:
        print(f"Warning: {file} is empty and will be skipped.")

if all_data:
    all_data_df = pd.concat(all_data, ignore_index=True)

    # Summary statistics for fixation duration
    fixation_durations = all_data_df['FixationDuration']
    mean_duration = fixation_durations.mean()
    median_duration = fixation_durations.median()
    std_duration = fixation_durations.std()

    print(f"Mean Fixation Duration: {mean_duration}")
    print(f"Median Fixation Duration: {median_duration}")
    print(f"Standard Deviation of Fixation Duration: {std_duration}")

    # Plotting fixation points
    plt.figure(figsize=(12, 8))
    plt.scatter(all_data_df['MappedFixationPointX'], all_data_df['MappedFixationPointY'], alpha=0.5)
    plt.gca().invert_yaxis()
    plt.title('Fixation Points on Apple Home Page')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    # Use eye tracking data for clustering
    eye_tracking_data = all_data_df.copy()

    # Initialize DBSCAN
    dbscan = DBSCAN(eps=30, min_samples=10)

    # Fit DBSCAN to the data
    clusters = dbscan.fit_predict(eye_tracking_data[['MappedFixationPointX', 'MappedFixationPointY']])

    # Remove noise points (-1 cluster)
    cleaned_data = eye_tracking_data[clusters != -1]


    web_page_image = mpimg.imread('Apple.png')
    plt.figure(figsize=(12, 8))
    plt.imshow(web_page_image)
    plt.scatter(cleaned_data['MappedFixationPointX'], cleaned_data['MappedFixationPointY'], c=clusters[clusters != -1],
                cmap='viridis', alpha=0.5)
    plt.title('Eye Tracking Clusters on Apple Web Page')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()