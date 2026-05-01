from scripts_environment_wrapper import environment
import pandas as pd

# Load cleaned / filtered data

df1 = pd.read_parquet("data/raw/ck/posts_from_top_accounts_ck.parquet")
df2 = pd.read_parquet("data/raw/ck/top_level_replies_to_posts_from_top_accounts_ck.parquet")
#df3 = pd.read_parquet("data/raw/ck/quotes_of_posts_from_top_accounts_ck.parquet")

# correct timestamps to strings
df1['timestamp'] = df1['timestamp'].astype(str)
df2['timestamp'] = df2['timestamp'].astype(str)
#df3['timestamp'] = df3['timestamp'].astype(str)

#Mark samples origins
df1['sample_type'] = 'influencers'
df2['sample_type'] = 'replies'
#df3['sample_type'] = 'quotes'


df = pd.concat([df1, df2])

df.to_parquet(environment.RAW_FILE_PATH())