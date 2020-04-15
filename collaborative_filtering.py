import sys
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# # Cosine similarity
# def standardize(row):
#     new_row = (row - row.mean()) / (row.max() - row.min())
#     return new_row
# ratings_std = user_ratings.apply(standardize)
# item_similarity = cosine_similarity(ratings_std.T)
# item_sim_df = pd.DataFrame(item_similarity, index=user_ratings.columns, columns=user_ratings.columns)
# item_sim_df.head()


# In[53]:

def get_similarity_matrix(ratings, kind):
    # Pearson Correlation Coeff
    if kind == 'user':
        return ratings.T.corr(method='pearson')
    elif kind == 'item':
        return ratings.corr(method='pearson')

# Predicts item based rating given user and item
def avg_item_rating(item):
    item_ratings = user_ratings[item]
    item_ratings = item_ratings[item_ratings > 0]
    return np.average(item_ratings)
    
def item_based_cf(u, i, neigh_size, item_sim, user_ratings):
    # average rating of item i
    av = avg_item_rating(i)
    # get most similar items to item i
    item_ids = list(item_sim)
    item_sim = item_sim[i]
    # most similar items
    neigh_items = sorted(zip(item_sim, item_ids), reverse=True)
    r = av
    num = 0
    denom = 0
    ct = 0
    for sim, it in neigh_items:
        if sim == 0 or user_ratings[it][u] == 0:
            continue
        num += sim * (user_ratings[it][u] - avg_item_rating(it))
        denom += sim
        ct += 1
        if ct == neigh_size:
            break
    r += num/denom
    return r

# Predicts item based rating given user and item
def avg_user_rating(user):
    u_ratings = user_ratings.T[user]
    u_ratings = u_ratings[u_ratings > 0]
    return np.average(u_ratings)
    
def user_based_cf(u, i, neigh_size, user_sim, user_ratings):
    # average rating of item i
    av = avg_user_rating(u)
    # get most similar items to item i
    user_ids = list(user_sim)
    user_sim = user_sim[u]
    # most similar users
    neigh_users = sorted(zip(user_sim, user_ids), reverse=True)
    r = av
    num = 0
    denom = 0
    ct = 0
    for sim, ur in neigh_users:
        if sim == 0 or user_ratings[i][ur] == 0:
            continue
        num += sim * (user_ratings[i][ur] - avg_user_rating(ur))
        denom += sim
        ct += 1
        if ct == neigh_size:
            break
    r += num/denom
    return r

if __name__ == '__main__':
    filename = sys.argv[1]
    ns = int(sys.argv[2])
    u = int(sys.argv[3])
    i = int(sys.argv[4])

    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='\t', names=r_cols, encoding='latin-1')
    user_ratings = ratings.pivot_table(index=['user_id'], columns=['movie_id'], values='rating')
    # Remove movies that have less than 10 users who have rated it - improves quality of reco
    user_ratings = user_ratings.dropna(thresh=10, axis=1).fillna(0)

    user_sim = get_similarity_matrix(user_ratings, 'user')
    item_sim = get_similarity_matrix(user_ratings, 'item')

    urating = user_based_cf(u,i,ns,user_sim,user_ratings)
    irating = item_based_cf(u,i,ns,item_sim,user_ratings)

    print('Ground Truth Rating = ', user_ratings[i][u] if user_ratings[i][u] > 0 else 'NA' )
    print('User-based CF Rating = ', urating)
    print('Item-based CF Rating = ', irating)