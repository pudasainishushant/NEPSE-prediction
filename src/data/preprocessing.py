import pandas as pd
import numpy as np
import os
import glob
import csv

#import shutil


def cleancsv(source):
    try:
        data = pd.read_csv(source, parse_dates=True)
        print('ok')
    except (FileNotFoundError, IOError):
        print('Wrong file or file path.')
        return
    if data.empty:
        print('data empty')
        return

    # shutil.copyfile(source, destination + source)
    # print('ok')

    data = data.drop_duplicates(subset='Date', keep='first')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', drop=False, inplace=True)
    idx = pd.date_range(data.index.min(), data.index.max())
    indexed_data = data.reindex(index=idx, fill_value=np.nan)
    indexed_data = indexed_data.replace('0', np.nan)
    indexed_data = indexed_data.fillna(method='ffill')
    indexed_data = indexed_data.drop('Date', 1)
    print(indexed_data)
    indexed_data.to_csv(source, index_label='Date')
    print('job done')
    # #filename = destination + '/' + source
    # with open(destination, 'w') as f:
    #     indexed_data.to_csv(f, index_label='Date')


def calcopening(source):
    try:
        data = pd.read_csv(source, index_col=0, parse_dates=True)
    except (FileNotFoundError, IOError):
        print('Wrong file or file path.')
        return
    if data.empty:
        return

    data['Opening Price'] = data['Closing Price'].shift(1)
    # The Opening Price must be adjusted so that it is smaller than Maximum Price
    # and larger than Minimum Price
    data['Maximum Price'] = data[['Opening Price', 'Maximum Price', 'Minimum Price', 'Closing Price']].max(axis=1)
    data['Minimum Price'] = data[['Opening Price', 'Maximum Price', 'Minimum Price', 'Closing Price']].min(axis=1)
    data.set_value(data.index[0], 'Opening Price', data.get_value(data.index[0], 'Closing Price'))
    data.to_csv(source, index=True)
    print('complete')


def cleanall(source):

    os.chdir(source)
    for file in glob.glob("*.csv"):
        filename = os.path.basename(file)
        print('Cleaning ' + filename + '...\n')
        cleancsv(filename)


def applyfunc(func, source, *args, **kwargs):

   # os.chdir(source)
    for file in glob.glob("*.csv"):
        filename = os.path.basename(file)
        func(file, *args, **kwargs)


if __name__ == "__main__":
    cleanall('./data/')
    applyfunc(calcopening, './data/')
