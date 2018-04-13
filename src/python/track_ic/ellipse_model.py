import numpy as np
import track_ic.ellipse_lib as el

class EllipseModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.

    """
    #def __init__(self,input_columns,output_columns,debug=False):
    #    self.input_columns = input_columns
    #    self.output_columns = output_columns
    #    self.debug = debug
    def fit(self, data):
        tp_data = np.transpose(data)
        tp_data = np.array([tp_data[1], tp_data[0]])
        lsqe = el.LSqEllipse()
        #print("fit: ", data)
        #print("fit tp: ", tp_data)
        lsqe.fit(tp_data)
        #print("ellipse: ", lsqe.parameters())
        return lsqe.parameters()

    def get_error( self, data, model):
        #print("error: ")
        #print(data)
        #print(model)
        return -1

