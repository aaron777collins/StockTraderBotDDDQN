import pandas as pd

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Calculate Bollinger Bands
    window_size = 20
    df['SMA'] = df['Close'].rolling(window=window_size).mean()
    df['STD'] = df['Close'].rolling(window=window_size).std()
    df['Upper_Band'] = df['SMA'] + (df['STD'] * 2)
    df['Lower_Band'] = df['SMA'] - (df['STD'] * 2)

    # Normalize the data
    features_to_normalize = ['Close', 'Upper_Band', 'Lower_Band', 'Open', 'High', 'Low']
    df[features_to_normalize] = df[features_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Fill NaN values in Bollinger Bands with first non-NaN values
    df[['SMA', 'STD', 'Upper_Band', 'Lower_Band']] = df[['SMA', 'STD', 'Upper_Band', 'Lower_Band']].fillna(method='bfill')

    # drop date column
    df = df.drop(columns=['Date'])

    return df
