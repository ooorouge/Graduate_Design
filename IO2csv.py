import csv
import numpy as np
from data_generate import *

def Output2csv(scale, deadtime_arc, irritation, buoyant_for_test=(10*math.pi/180)):
    '''
    :param irratation: numbers of sets of data
    :return: None
    '''
    open_csv = open('Raw_data.csv', 'w', newline='')
    csv_write = csv.writer(open_csv, dialect='excel')
    for i in range(irritation):
        raw_data = DataSamples_Type2(scale, i / irritation * deadtime_arc, buoyant_for_test)
        _t = raw_data.extendData()
        csv_write.writerow(_t)

def ResultsOutput(test_preds, test_actuals, filename='results_visualization.csv', mode='a'):
    '''
    :param test_preds: output of predictions
    :param test_actuals: as it says
    :param filename: filename and path
    :param mode: IO mode such as 'w' or 'a' or whatever
    :return: None
    '''
    with open(filename, mode, newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        for i, j in enumerate(test_preds):
            _t = [test_actuals[i], j]
            csv_write.writerow(_t)

def ReadFromCsv(filename='input_data_withSliding.csv'):
    '''
    :param filename: filename and path
    :return:
            x_vals: Data with features and labels
            x_vals_squeeze: x_vals without labels
            y_vals_squeeze: only labels sorted
    '''
    x_vals = np.zeros((500, 43), dtype=np.float)
    y_vals = np.zeros((500, 43), dtype=np.float)
    x_vals_squeeze = np.zeros((500, 42), dtype=np.float)
    y_vals_squeeze = np.zeros(500, dtype=np.float)
    i = 0
    len_of_1_line = 0
    with open(filename, 'r', newline='') as f:
        csv_files = csv.reader(f)
        for i, line in enumerate(csv_files):
            x_vals[i] = np.array(line)

    for i in range(500):
        y_vals_squeeze[i] = x_vals[i][42]
        x_vals_squeeze[i] = x_vals[i][:42]

    return x_vals, x_vals_squeeze, y_vals_squeeze