def get_movie_id(movies, title, year=None, title_col='title', year_col='year'):
    """get movie_id using a movie name

    Args:
        movies (pd.DataFrame): movies DataFrame cols: title | year | *genres
        title (string): name of the movie
        year (int, optional): year of the movie. Defaults to None.
        title_col (str, optional): title column name. Defaults to 'title'.
        year_col (str, optional): year column name. Defaults to 'year'.

    Returns:
        int: id of the movie
    """    
    res = movies[movies[title_col] == title]
    if year:
        res = res[res[year_col] == year]

    if len(res) > 1:
        print("Ambiguous: found")
        print(f"{res['title']} {res['year']}")
    elif len(res) == 0:
        print('not found')
    else:
        return res.index[0]

def get_movie_name(movies, index, title_col='title'):
    """get movie name using a movie id

    Args:
        movies (pd.DataFrame): movies DataFrame cols: title | year | *genres
        index (int): id of the movie
        title_col (str, optional): title column name. Defaults to 'title'.

    Returns:
        string: name of the movie
    """    
    return movies.iloc[index][title_col]

def get_movie_year(movies, index, year_col='year'):
    """get movie year using a movie id

    Args:
        movies (pd.DataFrame): movies DataFrame cols: title | year | *genres
        index (int): id of the movie
        year_col (str, optional): year column name. Defaults to 'year'.

    Returns:
        int: year of the movie
    """    
    return movies.iloc[index][year_col]
