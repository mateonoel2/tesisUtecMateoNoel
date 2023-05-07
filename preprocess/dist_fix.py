def dist_fix(df):
    mean = df['distance'].mean()
    df.loc[:, 'distance'] = mean
    return df