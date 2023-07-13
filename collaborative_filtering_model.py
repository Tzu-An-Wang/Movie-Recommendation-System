import torch


class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        users = self.users[idx]
        movies = self.movies[idx]
        ratings = self.ratings[idx]

        return {"users":torch.tensor(users, dtype=torch.long),
                "movies":torch.tensor(movies, dtype=torch.long),
                "ratings":torch.tensor(ratings, dtype=torch.long)}
    
