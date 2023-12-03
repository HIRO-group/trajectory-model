import math
import numpy as np
from trajectory_model.spill_free.constants import BIG_DIAMETER_B, BIG_HEIGHT, BIG_DIAMETER_U, BIG_FILL_80, BIG_FILL_30, SMALL_DIAMETER_B, SMALL_HEIGHT, SMALL_DIAMETER_U, SMALL_FILL_80, SMALL_FILL_50, SHORT_TUMBLER_DIAMETER_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_DIAMETER_U, SHORT_TUMBLER_FILL_30, SHORT_TUMBLER_FILL_70, TALL_TUMBLER_DIAMETER_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_DIAMETER_U, TALL_TUMBLER_FILL_50, TALL_TUMBLER_FILL_80, TUMBLER_DIAMETER_B, TUMBLER_HEIGHT, TUMBLER_DIAMETER_U, TUMBLER_FILL_30, TUMBLER_FILL_70, WINE_DIAMETER_B, WINE_HEIGHT, WINE_DIAMETER_U

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
    h_c = TUMBLER_HEIGHT
    h_w = 0.1 * h_c
    r_u = TUMBLER_DIAMETER_U/2
    r_b = TUMBLER_DIAMETER_B/2

    in_meters = convert_inches_to_meters(np.array([h_c, h_w, r_u, r_b]))
    tilt_max = calc_max_tilt_angle(*in_meters)

    print("max tilt degree is: ", tilt_max)
