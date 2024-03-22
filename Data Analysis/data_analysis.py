import pandas as pd 
import os 


script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))

from data_analysis_functions import * 


#Read file
df = pd.read_csv('analysis.csv')

# Drop or generate columns
# Convert columns to datetime
date_cols = ['day', 'auction_start', 'auction_end', 'bidder_start', 'bidder_end']
df[date_cols] = df[date_cols].apply(pd.to_datetime)
#Drop bidder source because it has only 1 value
df=df.drop(columns=['bidder_source'])

# Size of bid 
df['bidder_size'] = df['bidder_height']*df['bidder_width']
df = df.drop(columns=['bidder_height', 'bidder_width'])



# Create DataFrame for each bidder_status type
df_bid = df[df['bidder_status'] == 'bid']
df_no_bid = df[df['bidder_status'] == 'noBid']
df_timeout = df[df['bidder_status'] == 'timeout']

#Create heatmap after encoding categorical columns to numeric
df_encoded,_=encode_dataframe(df)
plot_heatmap(df_encoded,bidder_status='all')

# df_encoded_bid,_=encode_dataframe(df_bid)
# plot_heatmap(df_encoded_bid,bidder_status='bid')

# df_encoded_no_bid,_=encode_dataframe(df_no_bid)
# plot_heatmap(df_encoded_no_bid,bidder_status='no bid')

# df_encoded_timeout,_=encode_dataframe(df_timeout)
# plot_heatmap(df_encoded_timeout,bidder_status='timeout')

#Plot top 10 categories 
plot_top_categories(df_bid,'country')
plot_top_categories(df_bid, 'bidder_media_type')


plot_top_per_categorical_group(df,'country','bidder_status','timeout')
plot_top_per_categorical_group(df,'country','bidder_status','bid')

# Plot histogram to visualize distribution of auction durations
plt.figure(figsize=(10, 6))
sns.histplot(df_bid['auction_ttl'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Auction Durations')
plt.xlabel('Auction Duration (milliseconds)')
plt.ylabel('Frequency')
plt.show()
plt.savefig(f'auction_ttl_distribution.png', bbox_inches='tight')


# Correlation analysis
correlation = df_bid[['auction_ttl', 'bidder_cpm']].corr(method='pearson')
print("Correlation between Auction Duration and Bidder CPM:")
print(correlation)


t_statistic, p_value = hypothesis_testing(df_bid['auction_ttl'], df_no_bid['auction_ttl'], 'Bid', 'No Bid')
t_statistic, p_value = hypothesis_testing(df_bid['auction_ttl'], df_timeout['auction_ttl'], 'Bid', 'Timeout')



# 1. Compare auction durations between different device types or operating systems
compare_groups(df_bid, 'device', 'auction_ttl', 'Comparison of Auction Durations Across Different Devices',
               'Device Type', 'Auction Duration (milliseconds)')



# Perform statistical tests
# compare auction durations between two consent groups using t-test
consent_group1 = df[df['consent'] == True]['auction_ttl']
consent_group2 = df[df['consent'] == False]['auction_ttl']

t_statistic, p_value = stats.ttest_ind(consent_group1, consent_group2)
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")