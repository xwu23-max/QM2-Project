import pandas as pd


def clean_london_metadata(input_file, output_file):

    df = pd.read_csv(input_file)


    target_cities = ['london', 'islington', 'kensington']
    london_df = df[df['city'].isin(target_cities)].copy()


    london_df['line'] = london_df['line'].fillna('Control/Unknown')
    london_df = london_df.rename(columns={'line': 'station_name'})


    london_df['surface_material'] = london_df['surface_material'].fillna('Unknown')
    london_df['surface_material'] = london_df['surface_material'].replace('-', 'Unknown')


    selected_columns = [
        'uuid',
        'city',
        'station_name',
        'latitude',
        'longitude',
        'surface_material',
        'sample_type',
        'num_reads',
        'control_type'
    ]

    london_cleaned = london_df[selected_columns]

    london_cleaned.to_csv(output_file, index=False)
    print(london_cleaned.head())
    print(london_cleaned.info())



if __name__ == "__main__":
    input_csv = 'complete_metadata.csv'
    output_csv = 'london_subway_metadata_cleaned.csv'

    clean_london_metadata(input_csv, output_csv)