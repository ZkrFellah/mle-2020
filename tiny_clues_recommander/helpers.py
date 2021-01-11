def get_movie_id(movies, title, year=None):
    """get movie_id using a movie name

    Args:
        movies (pd.DataFrame): movies DataFrame cols: title | year | *genres
        title (string): name of the movie
        year (int, optional): year of the movie. Defaults to None.

    Returns:
        int: id of the movie
    """    
    res = movies[movies['title'] == title]
    if year:
        res = res[res['year'] == year]

    if len(res) > 1:
        print("Ambiguous: found")
        print(f"{res['title']} {res['year']}")
    elif len(res) == 0:
        print('not found')
    else:
        return res.index[0]

def get_movie_name(movies, index):
    """get movie name using a movie id

    Args:
        movies (pd.DataFrame): movies DataFrame cols: title | year | *genres
        index (int): id of the movie
    Returns:
        string: name of the movie
    """    
    return movies.iloc[index].title

def get_movie_year(movies, index):
    """get movie year using a movie id

    Args:
        movies (pd.DataFrame): movies DataFrame cols: title | year | *genres
        index (int): id of the movie
    Returns:
        int: year of the movie
    """    
    return movies.iloc[index].year
