import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mtick


#initial configuration 
sns.set(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None

csv_files = ["GBvideos.csv", "USvideos.csv"]

dfs = []
#read csv files and append them to a list of data frames
for f in csv_files:
    tag = f[0:2]
    print(tag)
    file_path = Path("Data Analysis 3") / f
    temp_df = pd.read_csv(file_path)
    temp_df['country'] = tag
    dfs.append(temp_df)

#clean the data by converting columns to strings to be unified and avoid unexpected behavior
#along with this filling all NaN values in the tags column with "N/A" instead for improved stability
def clean_dataframe(df):
    columns = ['video_id', 'title', 'channel_title', 'category_id', 'tags', 'thumbnail_link']
    
    df['tags'] = df['tags'].fillna('N/A')
    for col in columns:
        df[col] = df[col].astype(str)

    return df


cleaned_dataframes = [clean_dataframe(df) for df in dfs]

#creates a dataframe of all the missing data stats for both countries
#data frame contains country code, the total missing entries, and the percentage of entires that are missing 
def analyze_missing_data(df_list, country_codes):
    results = []
    for df, code in zip(df_list, country_codes):
        missing_total = df.isna().sum().sum()
        print(df.size)
        missing_percent = missing_total/df.size

        row = {
        'country_code': code,
        'missing_count': missing_total,
        'missing_percent': missing_percent
        }
        results.append(row)
    
    missing_report = pd.DataFrame(results)
    return missing_report


#generate missing report
country_codes = ["GB", "US"]
missing_report = analyze_missing_data(dfs, country_codes)

#combine and back up data frames for future analysis
combined_df = pd.concat(cleaned_dataframes, ignore_index=True)
backup_df = combined_df.copy()

#delete duplicates of video ids to not have repeat data
combined_df = combined_df.drop_duplicates(subset='video_id', keep='first')
combined_df = combined_df.set_index('video_id')


#Feature engineering
#Add multiple columns that give insights on the videos and can be used for other analysis
combined_df['like_ratio'] = combined_df['likes'] / combined_df['dislikes']
combined_df['engagement_rate'] = (combined_df['likes'] + combined_df['dislikes'] + combined_df['comment_total']) / combined_df['views']
combined_df['title_length'] = combined_df['title'].str.len()
combined_df['title_word_count'] = combined_df['title'].str.split().apply(len)
combined_df['title_has_exclamation'] = combined_df['title'].str.contains('!')
combined_df['tags_count'] = combined_df['tags'].str.split('|').apply(len)

print("\nNew features created:")
print(combined_df[['like_ratio', 'engagement_rate', 
                    'title_length', 'tags_count']].describe())


#create a figure with subplots to display the data
plt.figure(figsize=(18, 12))
#subplot 1: Makes a dot plot of views and likes. From 1000 random samples from all countries
plt.subplot(1,2,1)
sample_df = combined_df.sample(1000, random_state=42)
countries = sample_df['country'].unique()

for c in countries:
    subset = sample_df[sample_df['country'] == c]
    plt.scatter(subset['views'], subset['likes'], label=c, alpha=0.5)

plt.xlabel("Views")
plt.ylabel("Likes")
plt.title("Views vs Likes by Country")
plt.legend()

#Subplot 2: Shows the engagement rate by country
plt.subplot(1,2,2)
sns.boxplot(data=combined_df, x='country', y='engagement_rate')
plt.title("Engagement Rate by Country")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("plot.png")


#Generate country analysis on the average views and dislikes
#seperated into two bar graphs because the scales are so off dislikes is invisible if it is just in one
def analyze_country(df, country_code):
    country_df = df[df['country'] == country_code]
 
    views_mean = country_df['views'].mean()
    dislikes_mean = country_df['dislikes'].mean()
 
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 
    axes[0].bar(['Average Views'], [views_mean], color='steelblue')
    axes[0].set_title(f"{country_code} - Average Views")
    axes[0].set_ylabel("Count")
    axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
 
    axes[1].bar(['Average Dislikes'], [dislikes_mean], color='salmon')
    axes[1].set_title(f"{country_code} - Average Dislikes")
    axes[1].set_ylabel("Count")
    axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
 
    plt.suptitle(f"{country_code} - Average Views vs Dislikes")
    plt.tight_layout()
    plt.savefig(f"{country_code}_analysis.png")
    plt.close()
 
    return


#Analyzes the relationship between how long a title is compared to how many views they get and seperates by country
def analyze_title_length_vs_views(df):
    df['title_length'] = df['title'].str.len()
    df['title_len_bin'] = (df['title_length'] // 5) * 5

    grouped = df.groupby(['country', 'title_len_bin'])['views'].mean().reset_index()
    grouped['views'] = grouped['views'].astype(int)
    plt.figure(figsize=(10, 6))

    for c in grouped['country'].unique():
        data = grouped[grouped['country'] == c]
        plt.plot(data['title_len_bin'], data['views'], marker='o', label=c)

    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.xlabel("Title Length (binned)")
    plt.ylabel("Average Views")
    plt.title("Title Length vs Views by Country")
    plt.legend()
    plt.grid(True)

    plt.savefig("title_length_vs_views.png")

    return


analyze_title_length_vs_views(combined_df)
for country in ['US', 'GB']:
    analyze_country(combined_df, country)

#save cleaned data to csv file 
combined_df.to_csv("cleaned_youtube_trending_data.csv", index=False)

top_categories = combined_df['category_id'].value_counts().head(5)

#plot to show top categories based on how many videos there are on it
plt.figure(figsize=(8, 5))
plt.bar(top_categories.index.astype(str), top_categories.values)

plt.xlabel("Category ID")
plt.ylabel("Number of Videos")
plt.title("Top 5 Global Categories")

plt.savefig("top_categories.png")
plt.close()

print("Saved cleaned data to 'cleaned_youtube_trending_data.csv'")
print("Saved visualization to 'top_categories.png'")