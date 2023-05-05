def dist_fix(df):
    mean = df['distance'].mean()
    df.loc[:, 'distance'] = mean
    print("dist 1")
    return df