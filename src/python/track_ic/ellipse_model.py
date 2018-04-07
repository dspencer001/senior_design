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
        print(data)
        return 1
    def get_error( self, data, model):
        print(data)
        print(model)
        return -1
 