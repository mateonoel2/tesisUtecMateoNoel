def dist_fix(df):
    mean = df['distance'].mean()[0]
    df.loc[:, 'distance'] = mean
    return df