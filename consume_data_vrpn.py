import vrpn
import numpy as np

class VRPNclient:
   """
       This client has only been tested in Python3.X, 2.7
   """
   def callback(self, userdata, data):
       self.tracked = True
       self.data_read = {userdata: data}

   def __init__(self, tracker_name, hostID):
        self.tracker_name = tracker_name
        self.hostID= hostID
        self.tracked = False
        self.data_read = None

        self.tracker = vrpn.receiver.Tracker(tracker_name + "@" + hostID)
        self.tracker.register_change_handler(self.tracker_name, self.callback, "position")
        self.info = []
        print("created class!")

   def sample_data(self):
        print("what?")
        self.tracker.mainloop()

   def get_observation(self):
       while not self.tracked:
           self.sample_data()
       self.info = []
       self.info += list(self.data_read[self.tracker_name]['position'])
       q = list(self.data_read[self.tracker_name]['quaternion'])
       # if you want to explore the variable above, just dir() to see the keys
       self.info += q
       self.tracked = False
       return self.info

if __name__=='__main__':
   import time
   C = VRPNclient("DHead", "tcp://192.168.50.26:1510")
   while True:
        start = time.time()
        print("head: ", C.get_observation()) # collect a single observation
        elapsed = time.time() - start
        print("vrpn elapsed: ", 1./elapsed, " Hz")