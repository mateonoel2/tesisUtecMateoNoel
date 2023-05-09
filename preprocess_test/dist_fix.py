def dist_fix(df):
    df.loc[:,'distance'] = df['distance'].round(decimals=-1)
    mode = df['distance'].mode()[0]
    df = df[(df['distance'] > mode - 10) & (df['distance'] < mode + 10)]
    df.loc[:, 'distance'] = mode
    return df