import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Step 1: Create a sample dataframe
def create_dataframe():
    data = {
        "review": [
            "This product is amazing",
            "Very bad experience",
            "Good quality and nice design",
            "Worst product ever",
            "Excellent and fantastic item",
            "Excellent item",
            "New product"
            
        ]
    }

    df = pd.DataFrame(data)
    return df


# Step 2: Save dataframe to CSV
def save_dataframe(df):

    os.makedirs("data", exist_ok=True)

    df.to_csv("data/data.csv", index=False)
    print("data.csv saved in 'data' folder.")


# Step 3: Load data.csv, vectorize, and create 'K' new columns
def process_data(k):

    # Load the saved dataframe
    df = pd.read_csv("data/data.csv")

    # Apply vectorization
    vectorizer = CountVectorizer(max_features=k)

    vectorized_data = vectorizer.fit_transform(df["review"])
    feature_names = vectorizer.get_feature_names_out()

    # Create dataframe with K new columns
    vectorized_df = pd.DataFrame(
        vectorized_data.toarray(),
        columns=feature_names
    )

    processed_df = pd.concat([df, vectorized_df], axis=1)

    # Save processed dataframe
    processed_df.to_csv("data/processed_data.csv", index=False)

    print(f"processed_data.csv saved in 'data' folder with {k} new columns.")

    return processed_df


# Main execution
if __name__ == "__main__":

    # Step 1
    df = create_dataframe()

    # Step 2
    save_dataframe(df)

    # Step 3
    k = 3   # Replace with desired number of features
    processed_df = process_data(k)

    print(processed_df.head())