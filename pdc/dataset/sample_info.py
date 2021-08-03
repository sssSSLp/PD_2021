import codecs
import pandas as pd


def read_raw_sample_info(background_path):
    with codecs.open(background_path, 'r', 'shift_jis', 'ignore') as file:
        df = pd.read_table(file, index_col=0, header=0, delimiter=',')
    return df


def read_sample_info(background_path):
    df = read_raw_sample_info(background_path)
    labels = {
        "Class": ['HL', 'PD'],
        "Trial_batch": ['n15', 'n50'],
        "gender": ['F', 'M']
    }
    transformed_df = df.copy(deep=True)
    for column, values in labels.items():
        for i, v in enumerate(values):
            transformed_df.loc[df[column] == v, column] = i
    extracted_df = transformed_df[['Class', 'Trial_batch', 'gender', 'age', 'Yahr']]
    extracted_df.columns = ['Class', 'TrialBatch', 'Sex', 'Age', 'Yahr']

    pd_drug_columns = df.columns[21:34]
    tmp_df = df[pd_drug_columns].copy(deep=True)
    tmp_df = tmp_df.fillna(0.0)

    s = (tmp_df > 0.0).any(axis=1)
    extracted_df['Dosing'] = 0
    extracted_df.loc[s, 'Dosing'] = 1

    new_index = []
    for i in extracted_df.index:
        if i.startswith('PD'):
            new_index.append(i)
        else:
            p = i.find('_')
            prefix = i[:p]
            identification = i[p + 1:]
            new_index.append(f'{prefix}{identification}')
    extracted_df.index = new_index
    return extracted_df
