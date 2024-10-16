import math
import numpy as np
from trajectory_model.process_data.containers import BasicGlass, WineGlass, FluteGlass, RibbedCup, TallCup, CurvyWineGlass

def convert_inches_to_meters(inches):
    return inches * 0.0254

def convert_to_degrees(radians):
    return radians * 180 / math.pi

def calc_max_tilt_angle(h_c, h_w, r_u, r_b):
    r_2 = r_u - r_b
    theta_1 = np.arctan(r_2/h_c)
    r_1 = np.tan(theta_1) * (h_c - h_w)
    r_w = r_u - r_1

    r_w_prime = (h_c*r_u*r_b - h_c*r_2*r_b + h_w*r_2*r_w + h_w*r_2*r_b)/(h_c*r_2 + h_c*r_b)
    h_1 = h_c * (r_u - r_w_prime) / r_2

    if r_w_prime > r_b:
        print("Case 1 occured.")
        theta_t = np.arctan(h_1/(r_u + r_w_prime))
        return convert_to_degrees(theta_t)
    
    elif r_w_prime <= r_b:
        print("Case 2 occured.")
        r_b_prime = 2 * h_w * (r_w + r_b) / h_c
        theta_t = np.arctan(h_c/(r_2 + r_b_prime))
        return convert_to_degrees(theta_t)

if __name__ == "__main__":
    container = BasicGlass(BasicGlass.low_fill)

    h_c = container.height
    h_w = container.height * container.fill_level
    r_u = container.diameter_u/2
    r_b = container.diameter_b/2

    in_meters = convert_inches_to_meters(np.array([h_c, h_w, r_u, r_b]))
    tilt_max = calc_max_tilt_angle(*in_meters)

    print("Max container tilt degree is: ", tilt_max)
