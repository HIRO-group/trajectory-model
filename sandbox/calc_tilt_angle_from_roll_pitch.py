import math

def calc_tilt_angle(roll, pitch):

    roll = math.radians(roll)
    pitch = math.radians(pitch)

    tilt_angle = math.acos(math.cos(roll) * math.cos(pitch))

    tilt_angle = math.degrees(tilt_angle)
    print("Tilt angle: ", tilt_angle)
    return tilt_angle


def calc_tilt_angle_v2(roll, pitch):
    roll = math.radians(roll)
    pitch = math.radians(pitch)

    theta_t_squared = math.tan(roll) * math.tan(roll) + math.tan(pitch) * math.tan(pitch)
    tilt_angle = math.atan(math.sqrt(theta_t_squared))
    tilt_angle = math.degrees(tilt_angle)

    print("Tilt angle v2: ", tilt_angle)

if __name__ == "__main__":
    roll = 38
    pitch = 25
    calc_tilt_angle(roll, pitch)
    calc_tilt_angle_v2(roll, pitch)