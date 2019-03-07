import pandas as pd
#id,belongs_to_collection,budget,genres,homepage,imdb_id,original_language,original_title,overview,popularity,poster_path,production_companies,production_countries,release_date,runtime,spoken_languages,status,tagline,title,Keywords,cast,crew,revenue

df = pd.read_csv(train_file_name)

for index, row in df.iterrows():
    #df['belongs_to_collection_id']
    if not pd.isnull(row['genres']):
        row['genres_json'] = pd.read_json(row['genres'].replace("'", '"'))
    else:
        row['genres_json'] = ""

