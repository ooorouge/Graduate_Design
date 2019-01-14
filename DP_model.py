import Globalparameters
from IO2csv import Output2csv

if __name__ == "__main__":
    scale = Globalparameters.scale
    #raw_data = [0 for i in range(scale)]
    deadtime_deg = Globalparameters.deadtime_deg
    deadtime_arc = Globalparameters.deadtime_arc
    buoyant_for_test = Globalparameters.buoyant_for_test
    Output2csv(scale, deadtime_arc, 200, buoyant_for_test)