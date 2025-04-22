import numpy as np
import pandas as pd

"""
Data consists of 3 tables: info about cars, drivers and rides.
We need to predict car breakdowns.
"""


def add_simple_features(df):
    # format for 2020-6-20 2:14	 in col 'fix_date'
    df['fix_date'] = pd.to_datetime(df['fix_date'], format="%Y-%m-%d %H:%M", errors='coerce')

    df['distance_delta'] = df['speed_avg'] * df['ride_duration'] - df['distance']
    df['log_distance_delta'] = df['distance_delta'].apply(lambda x: np.log(np.abs(x)) * np.sign(x))
    return df


def get_rides_n_drivers_features(
        df,
        cols=(
            'user_rating', 'user_time_accident', 'rating', 'speed_max', 'speed_avg',
            'user_ride_quality', 'deviation_normal', 'log_distance_delta'
        )
):
    # group df by day
    df_avg_by_day = df.groupby(['car_id', 'ride_date'], as_index=False).agg(
        **{f'{col}_avg_1d': (col, 'mean') for col in cols}
    )
    df_avg_avg = df_avg_by_day.groupby('car_id', as_index=False).agg(
        **{f'{col}_avg_avg': (f'{col}_avg_1d', 'mean') for col in cols},
        **{f'{col}_avg_std': (f'{col}_avg_1d', 'std') for col in cols},
        **{f'{col}_avg_min': (f'{col}_avg_1d', 'min') for col in cols},
        **{f'{col}_avg_max': (f'{col}_avg_1d', 'max') for col in cols},
    )
    print(df_avg_by_day.shape)
    print(df_avg_avg.shape)
    df_avg_few_days_stat = df_avg_by_day.groupby('car_id', as_index=False).agg(
        # for first/last 1 day
        deviation_normal_avg_last_1d=('deviation_normal_avg_1d', lambda x: x.iloc[-1]),
        user_ride_quality_avg_last_1d=('user_ride_quality_avg_1d', lambda x: x.iloc[-1]),
        deviation_normal_avg_first_1d=('deviation_normal_avg_1d', lambda x: x.iloc[0]),
        user_ride_quality_avg_first_1d=('user_ride_quality_avg_1d', lambda x: x.iloc[0]),
        # for first/last 3 days
        deviation_normal_avg_last_3d=('deviation_normal_avg_1d', lambda x: np.mean(x.iloc[-3:])),
        user_ride_quality_avg_last_3d=('user_ride_quality_avg_1d', lambda x: np.mean(x.iloc[-3:])),
        deviation_normal_avg_first_3d=('deviation_normal_avg_1d', lambda x: np.mean(x.iloc[:3])),
        user_ride_quality_avg_first_3d=('user_ride_quality_avg_1d', lambda x: np.mean(x.iloc[:3])),
    )
    res = df_avg_avg.merge(df_avg_few_days_stat, on='car_id', how='left')
    print(res.shape)

    return res


def get_rides_features(df, rides):
    if 'mean_rating' not in df.columns:
        f = lambda x: x.nunique()
        rides_df_gr = rides.groupby('car_id', as_index=False).agg(
            mean_rating=('rating', 'mean'),
            rating_min=('rating', 'min'),
            distance_sum=('distance', 'sum'),
            speed_max=('speed_max', 'max'),
            speed_max_p90=('speed_max', lambda x: np.percentile(x, 90)),
            speed_avg_p85=('speed_avg', lambda x: np.percentile(x, 90)),
            user_ride_quality_median=('user_ride_quality', 'median'),
            deviation_normal_median=('deviation_normal', 'median'),
            user_uniq=('user_id', f),
        )

        # add max shift and max shift index (aka day number)
        f1 = lambda x: np.max(x.diff()) if np.max(x.diff()) >= 5 else 0
        f2 = lambda x: np.argmax(x.diff())

        df_deviation_shift = rides.groupby(['car_id', 'ride_date'], as_index=False).agg(
            deviation_normal_avg_1d=('deviation_normal', 'mean'),
        ).sort_values(
            'ride_date'
        ).groupby('car_id', as_index=False).agg(
            deviation_normal_max_shift=('deviation_normal_avg_1d', f1),
            deviation_normal_day_max_shift=('deviation_normal_avg_1d', f2)
        )
        return df.merge(rides_df_gr, on='car_id', how='left').merge(df_deviation_shift, on='car_id', how='left')
    else:
        return df
