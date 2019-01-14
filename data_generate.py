import math
import matplotlib.pyplot as plt

class DataSamples:
    """for generate data that could be used"""
    def __init__(self, scale, deadtime):
        self.scale = scale
        self.deadtime = deadtime
        self.pi = math.pi
        self.revise = math.sin(self.deadtime)
        self.raw_data = [0 for i in range(self.scale)]

    def generateWithDeadzone(self):
        '''
        HALF WAVE
        '''
        for i in range(self.scale):
            temp = i*self.pi/self.scale
            if temp<self.deadtime or temp>self.pi - self.deadtime:
                self.raw_data[i] = 0
            else:
                self.raw_data[i] = math.sin(temp) - self.revise
        return self.raw_data

    def extendData(self):
        '''
        ONE PERIOD
        '''
        _t = self.generateWithDeadzone()
        mid_index = len(_t)
        _ex = [0 for i in range(2*mid_index)]
        for i in range(0,mid_index):
            _ex[i] = _t[i]
        for i in range(mid_index, 2*mid_index):
            _ex[i] = -_t[i - mid_index]
        return _ex

class DataSamples_Type2(DataSamples):
    '''峰值处有些改动'''
    def __init__(self, scale, deadtime, buoyant):
        super(DataSamples_Type2, self).__init__(scale, deadtime)
        self.buoyant = buoyant
        self.buoyant_value = math.sin(math.pi/2 - self.buoyant) - self.revise
        self.upperBound = math.pi / 2 + self.buoyant
        self.lowerBound = math.pi / 2 - self.buoyant


    def generate(self):
        mid_index = math.pi / 2
        k = 1                                    #crucial!!!!
        for i in range(self.scale):
            temp = i*self.pi/self.scale
            if temp<self.deadtime or temp>self.pi - self.deadtime:
                self.raw_data[i] = 0
            else:
                if temp > self.lowerBound and temp < self.upperBound:
                    self.raw_data[i] = 2*self.buoyant_value - k*(math.sin(temp) - self.revise)
                else:
                    self.raw_data[i] = math.sin(temp) - self.revise
        return self.raw_data

    def extendData(self):
        '''
        ONE PERIOD
        '''
        _t = self.generate()
        mid_index = len(_t)
        _ex = [0 for i in range(2*mid_index)]
        for i in range(0,mid_index):
            _ex[i] = _t[i]
        for i in range(mid_index, 2*mid_index):
            _ex[i] = -_t[i - mid_index]
        return _ex

def MultiDT_figure(scale, number, type='T1'):
    for i in range(number):
        deadtime_arc = i * math.pi / 180
        buoyant_for_test = 1.5 * i * math.pi / 180
        if type is 'T2':
            data = DataSamples_Type2(scale, deadtime_arc, buoyant_for_test)
        else:
            data = DataSamples(scale, deadtime_arc)
        raw_data_extend = data.extendData()
        _ = plt.plot(raw_data_extend, label = str(i) + 'deg line')
    plt.legend()
    plt.show()