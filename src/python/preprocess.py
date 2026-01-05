import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
class MovieLensDataLoader:
    """
    Main preprossing class to convert MovieLen raw data to model-ready format
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.sep = '::'
        self.engine = 'python'
        self.encoding = 'ISO-8859-1'
        
    
    def load_and_preprocess(self):
        """
        Executes the full pipeline
        """
        
        # Load data
        movies = self._load_movies()
        ratings = self._load_ratings()
        users = self._load_users()
        
        # Generate user-level statistical features
        user_stats = ratings.groupby('userId').agg({'rating': ['mean', 'count']})
        user_stats.columns = ['user_neam_rating', 'user_rating_count']
        user_stats.reset_index(inplace = True)
        
        # Label Encdoing
        user_le = LabelEncoder()
        movie_le = LabelEncoder()
        
        ratings['userId'] = user_le.fit_transform(ratings['userId'])
        ratings['movieId'] = movie_le.fit_transform(ratings['movieId'])
        user_stats['userId'] = user_le.transform(user_stats['userId'])
        # Save dimensions 
        self.field_dims = [len(user_le.classes_), len(movie_le.classes_)]
        
        # Final join
        # Merge all features into one large training table
        data = pd.merge(ratings, movies, on='movieId', how='left')
        data = pd.merge(data, user_stats, on='userId', how='left')
        
        # Fill missing values if any
        data.fillna(0, inplace=True)
        
        print(f"Log: Preprocessing complete. Final shape: {data.shape}")
        return data, self.field_dims
    
    def _load_movies(self) -> pd.DataFrame:
        path = os.path.join(self.data_path, 'movies.dat')
        df = pd.read_csv(path, sep=self.sep, names=['movieId', 'title', 'genres'],
                         engine=self.engine, encoding=self.encoding)
        # Genre one-hot encoding
        genres = df['genres'].str.get_dummies(sep='|')
        return pd.concat([df[['movieId']], genres], axis=1)
    
    def _load_ratings(self) -> pd.DataFrame:
        path = os.path.join(self.data_path, 'ratings.dat')
        return pd.read_csv(path, sep=self.sep, names=['userId', 'movieId', 'rating', 'timestamp'],
                           engine=self.engine, encoding=self.encoding)
        
    def _load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_path, 'users.dat')
        return pd.read_csv(path, sep=self.sep, names=['userId', 'gender', 'age', 'occupation', 'zip'],
                           engine=self.engine, encoding=self.encoding)
        
        
if __name__ == "__main__":
    # Ensure this matches your data/raw/ml-1m structure
    base_path = os.path.join('data', 'raw', 'ml-1m')
    loader = MovieLensDataLoader(base_path)
    final_df, dims = loader.load_and_preprocess()
    print(f"\nField Dimensions (User, Movie): {dims}")
    print("\nFinal Data Preview (First 5 rows):")
    print(final_df.head())