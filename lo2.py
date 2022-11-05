import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

state_data=['kashmir.csv', 'kerala.csv', 'ladakh.csv', 'arunachal.csv', 'andhra.csv', 'haryana.csv', 'Himachal.csv', 'manipur.csv', 'Meghalaya.csv', 'Sikkim.csv', 'Assam.csv', 'Bihar.csv', 'Jharkhand.csv', 'Mizoram.csv', 'Tamil Nadu.csv', 'Telangana.csv']

def tfidflk(csvf, x):
    
    metadata = pd.read_csv(csvf, low_memory=False)

    # print(metadata.head(5))
    # print(metadata['Unit'].head(10))             #Print plot overviews of the first 5 movies.

    tfidf = TfidfVectorizer(stop_words='english')
    metadata['Item'] = metadata['Item'].fillna('')
    tfidf_matrix = tfidf.fit_transform(metadata['Item'])
    # print(tfidf_matrix.shape)

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # print(cosine_sim.shape)
    # print(cosine_sim[1])

    indices = pd.Series(metadata.index, index=metadata['Item']).drop_duplicates()
    # indices = pd.Series(metadata.index, index=metadata['Item'])
    # indices=indices.drop_duplicates() 

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

    con = int(list(get_recommendations("Oranges, Mandarines"))[0].split(" ")[0]) - x

    if(con>0):
        print("The State/Regiment is short of resources by", con, "Tonnes\n")
    else:
        print("The State/Regiment is efficient of resources by", abs(con), "Tonnes\n")

    


for i in state_data:
    print("Enter the state data of",i[0:-4:],":")
    a=int(input("->"))
    tfidflk('states\\'+i, a)
    #print(i, "\n")

