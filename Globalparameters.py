import math

'''
    :param scale: Resolution of input data
    :param deadtime_deg: deadtime defined by degree
    :param deadtime_acr: deadtime defined by arcs
    :param buoyant_for_test: fluctuations on the peak of the wave 
    TODO: Full-connected layers output dims/ 
                         batch size/ 
                         convolution size/
                         max pool size/
                         stride size/
        AND their relationships with scale or desired dims of output features
        
    ALL in this source file
'''
scale = 25
deadtime_deg = 5
deadtime_arc = deadtime_deg * math.pi / 180
buoyant_for_test = 10 * math.pi / 180