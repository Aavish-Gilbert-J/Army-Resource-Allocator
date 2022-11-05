import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv('loc3.csv', low_memory=False)

# print(metadata.head(5))
# print(metadata['Unit'].head(10))             #Print plot overviews of the first 5 movies.

tfidf = TfidfVectorizer(stop_words='english')
metadata['Item'] = metadata['Item'].fillna('')
tfidf_matrix = tfidf.fit_transform(metadata['Item'])
# print(tfidf_matrix.shape)



cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# print(cosine_sim.shape)
# print(cosine_sim[1])

# indices = pd.Series(metadata.index, index=metadata['Item']).drop_duplicates(subset=['Item'])
indices = pd.Series(metadata.index, index=metadata['Item'])
indices=indices.drop_duplicates()

#print(indices[:100])   #cosine coordinates



# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title,  cosine_sim = cosine_sim):
    
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all units with that unit
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the units based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar units
    sim_scores = sim_scores[1:2]

    # Get the unit indices
    movie_indices = [i[0] for i in sim_scores]

    #print(metadata(pd.iloc[movie_indices]))

    # Return the top 10 most similar units
    return metadata['Unit'].iloc[movie_indices]

print(get_recommendations("Oranges, Mandarines"))



