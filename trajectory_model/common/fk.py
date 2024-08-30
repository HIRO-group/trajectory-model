import numpy as np
import roboticstoolbox as rtb

def get_panda():
    E1 = rtb.ET.tz(0.333)
    E2 = rtb.ET.Rz()
    E3 = rtb.ET.Ry()
    E4 = rtb.ET.tz(0.316)
    E5 = rtb.ET.Rz()
    E6 = rtb.ET.tx(0.0825)
    E7 = rtb.ET.Ry(flip=True)
    E8 = rtb.ET.tx(-0.0825)
    E9 = rtb.ET.tz(0.384)
    E10 = rtb.ET.Rz()
    E11 = rtb.ET.Ry(flip=True)
    E12 = rtb.ET.tx(0.088)
    E13 = rtb.ET.Rx(np.pi)
    E14 = rtb.ET.tz(0.107)
    E15 = rtb.ET.Rz()
    panda = E1 * E2 * E3 * E4 * E5 * E6 * E7 * \
        E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15
    return panda

def perform_fk(q_array, degrees=True):
    panda = get_panda()
    fk = panda.eval(q_array)
    return fk