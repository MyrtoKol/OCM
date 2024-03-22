from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy import stats

def encode_dataframe(df):
    """
    Encode categorical columns using LabelEncoder.

    Args:
        df (DataFrame): Input DataFrame.

    Returns:
        df_encoded (DataFrame): Encoded DataFrame.
        encoding_mappings (dict): Mapping of original labels to encoded values.
    """
    # Get non-numeric and non-timestamp columns
    non_numeric_cols = df.select_dtypes(include='object').columns

    # Initialize a dictionary to store mappings
    encoding_mappings = {}

    # Create a copy of the DataFrame
    df_encoded = df.copy()

    # Apply label encoding
    label_encoder = LabelEncoder()
    for col in non_numeric_cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
        # Store the mapping in the dictionary
        encoding_mappings[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Drop the specified columns
    columns_to_drop = ['day', 'auction_start', 'auction_end', 'bidder_start', 'bidder_end']
    df_encoded = df_encoded.drop(columns=columns_to_drop)

    return df_encoded, encoding_mappings

def plot_heatmap(df_encoded, bidder_status, threshold=0.3):
    """
    Plot a correlation heatmap of the encoded DataFrame.

    Args:
        df_encoded (DataFrame): Encoded DataFrame.
        bidder_status (str): Bidder status for plot title.
        threshold (float): Threshold for correlation annotation.

    Returns:
        None
    """
    # Plot heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(df_encoded.corr(), cmap='coolwarm', vmin=-1, vmax=1)

    # Add annotations
    corr = df_encoded.corr()
    for i in range(len(corr)):
        for j in range(len(corr)):
            if abs(corr.iloc[i, j]) > threshold:
                plt.text(j + 0.5, i + 0.5, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

    # Set title and display the plot
    plt.title(f'Correlation Heatmap of Encoded DataFrame for Bidder Status: {bidder_status}')
    plt.show()

    # Save the plot
    plt.savefig(f'Correlation Heatmap of Encoded DataFrame for Bidder Status_{bidder_status}.png', bbox_inches='tight')

def plot_top_categories(df, group_by_column, metric_column='bidder_cpm', n=10, ascending=False):
    """
    Plot top categories by average metric.

    Args:
        df (DataFrame): Input DataFrame.
        group_by_column (str): Column to group by.
        metric_column (str): Column for metric calculation.
        n (int): Number of top categories to plot.
        ascending (bool): Whether to sort in ascending order.

    Returns:
        None
    """
    # Group DataFrame by the specified column
    grouped_df = df.groupby(group_by_column)
    
    # Calculate the average metric for each group
    average_metric_by_group = grouped_df[metric_column].mean()

    # Sort the groups based on the average metric
    sorted_groups = average_metric_by_group.sort_values(ascending=ascending)

    # Select the top N categories
    top_n_categories = sorted_groups.tail(n)

    # Create a bar plot
    plt.figure(figsize=(10, 10))
    top_n_categories.plot(kind='bar', color='skyblue')
    plt.title(f'Top {n} Categories by Average {metric_column}')
    plt.xlabel(group_by_column)
    plt.ylabel(f'Average {metric_column}')
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig(f'barplot of top {n} {group_by_column} by Average {metric_column}.png', bbox_inches='tight')

def plot_top_per_categorical_group(df, group_by_column, categorical_column, categorical_column_value, min_events=10, n=10):
    """
    Plot top categories with the highest percentage of a specific value.

    Args:
        df (DataFrame): Input DataFrame.
        group_by_column (str): Column to group by.
        categorical_column (str): Categorical column to analyze.
        categorical_column_value (str): Value to filter by.
        min_events (int): Minimum number of events for a group to be considered.
        n (int): Number of top categories to plot.

    Returns:
        None
    """
    total_events_by_group = df[group_by_column].value_counts()
    valid_groups = total_events_by_group[total_events_by_group >= min_events].index
    df_filtered = df[df[group_by_column].isin(valid_groups)]
    grouped_by = df_filtered.groupby(group_by_column)

    counts_per_group = grouped_by[categorical_column].apply(lambda x: (x == categorical_column_value).sum())
    percentage = (counts_per_group / total_events_by_group.loc[valid_groups]) * 100
    sorted_percentage = percentage.sort_values(ascending=False)
    top_n = sorted_percentage.head(n)
    
    plt.figure(figsize=(10, 6))
    top_n.plot(kind='bar', color='skyblue')
    plt.title(f'Top {n} {group_by_column} entries with the Highest Percentage of {categorical_column_value} (with >= {min_events} entries)')
    plt.xlabel(group_by_column)
    plt.ylabel(f'Percentage of {categorical_column_value}')
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig(f'Top {n} {group_by_column} entries with the Highest Percentage of {categorical_column_value} (with >= {min_events} entries).png', bbox_inches='tight')


def hypothesis_testing(df1, df2, label1, label2):
    """
    Perform hypothesis testing between two DataFrame columns.

    Args:
    - df1 (DataFrame): DataFrame containing data for group 1
    - df2 (DataFrame): DataFrame containing data for group 2
    - label1 (str): Label for group 1
    - label2 (str): Label for group 2

    Returns:
    - t_statistic (float): T-statistic value
    - p_value (float): P-value
    """
    # Perform t-test
    t_statistic, p_value = ttest_ind(df1, df2, equal_var=False)
    
    # Print results
    print(f"Hypothesis Testing Results ({label1} vs. {label2}):")
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)
    
    return t_statistic, p_value

def compare_groups(df, x_column, y_column, title, x_label, y_label, rotation=None):
    """
    Compare the distribution of a numerical variable across different groups.

    Args:
    - df (DataFrame): DataFrame containing the data
    - x_column (str): Column representing the groups on the x-axis
    - y_column (str): Column representing the numerical variable on the y-axis
    - title (str): Title of the plot
    - x_label (str): Label for the x-axis
    - y_label (str): Label for the y-axis
    - rotation (int, optional): Rotation angle for x-axis labels (default: None)

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_column, y=y_column, data=df)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if rotation:
        plt.xticks(rotation=rotation)
    plt.show()
    plt.savefig(f'{title}.png', bbox_inches='tight')
