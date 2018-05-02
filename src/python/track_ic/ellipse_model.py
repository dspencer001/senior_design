import numpy as np
import track_ic.ellipse_lib as el
import math
import cv2

class EllipseModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.

    """
    def __init__(self, grad_x, grad_y, img):
        self.img = img
        self.orig_img = np.copy(img)
        self.angles = np.linspace(0, 2 * math.pi, 40)
        self.grad_x = grad_x
        self.grad_y = grad_y
    #    self.input_columns = input_columns
    #    self.output_columns = output_columns
    #    self.debug = debug

    def fit(self, data):
        self.recent_data = data

        tp_data = np.transpose(data)
        tp_data = np.array([tp_data[1], tp_data[0]])
        lsqe = el.LSqEllipse()
        #print("fit: ", data)
        #print("fit tp: ", tp_data)
        lsqe.fit(tp_data)
        #print("ellipse: ", lsqe.parameters())
        return lsqe

    def pt_error(self, coefficients, center, pt):
        (a, b, c, d, f, g) = coefficients
        x = pt[0] - center[0]
        y = pt[1] - center[1]
        grad_x = self.grad_x[pt[1]][pt[0]]
        grad_y = self.grad_y[pt[1]][pt[0]]

        X_x = np.array([2 * x, y, 0, 1, 0, 0])
        X_y = np.array([0, x, 2 * y, 0, 1, 0])
        V = np.array([a, b, c, d, f, g])

        g_mag = math.sqrt(V.dot(X_x)**2 + V.dot(X_y)**2)

        error = a * x**2 + 2 * b * x * y + c * y**2 + 2 * d * x + 2 * f * y + g
        error = error / g_mag

        return abs(error)

    def get_error(self, data, model):
        gradient_sum = 0
        center = model.center
        semimajor_radius = model.width
        semiminor_radius = model.height
        cos_angle = math.cos(model.phi)
        sin_angle = math.sin(model.phi)
        img_dimensions = self.grad_x.shape
        coefficients = model.coefficients()
        #print(model.parameters())

        error_func = lambda pt: self.pt_error(coefficients, center, pt)

        pt_err = np.array(list(map(error_func, data)))
        return pt_err

        #for pt in data:
        #    print(error_func(pt))
        #    x = pt[0] - center[0]
        #    y = pt[1] - center[1]
        #    grad_x = self.grad_x[pt[1]][pt[0]]
        #    grad_y = self.grad_y[pt[1]][pt[0]]

        #    X_x = np.array([2 * x, y, 0, 1, 0, 0])
        #    X_y = np.array([0, x, 2 * y, 0, 1, 0])
        #    V = np.array([a, b, c, d, f, g])

        #    g_mag = math.sqrt(V.dot(X_x)**2 + V.dot(X_y)**2)

        #    error = a * x**2 + 2 * b * x * y + c * y**2 + 2 * d * x + 2 * f * y + g
        #    error = error / g_mag

        #    print(error)
        #    #print(V)
        #    #print(res)

        #                #print(error)
        #    #print(error / res)

        return -1


#        for theta in self.angles:
#            x = center[0] + semimajor_radius * math.cos(theta)
#            y = center[1] + semiminor_radius * math.sin(theta)
#            nx = (x - center[0]) * semimajor_radius / semiminor_radius
#            ny = (y - center[1]) * semiminor_radius / semimajor_radius
#            mag = math.sqrt(nx * nx + ny * ny)
#            nx = nx / mag
#            ny = ny / mag
#
#            # rotate by ellipse angle
#            rot_x = x * cos_angle - y * sin_angle
#            rot_y = x * sin_angle + y * cos_angle
#            rot_nx = nx * cos_angle - ny * sin_angle
#            rot_ny = nx * sin_angle + ny * cos_angle
#
#            rot_point = np.round([rot_x, rot_y]).astype(int)
#
#            if(rot_point[0] < 0 or rot_point[1] < 0 or rot_point[0] >= img_dimensions[1] or rot_point[1] >= img_dimensions[0]):
#                continue
#
#            grad_x = self.grad_x[rot_point[1]][rot_point[0]]
#            grad_y = self.grad_y[rot_point[1]][rot_point[0]]
#
#            gradient_sum += min(rot_nx * grad_x + rot_ny * grad_y, 0)
#
#        if gradient_sum != 0:
#            return 10000.0 / ((-1.0) * gradient_sum)
#
#        return 10000

