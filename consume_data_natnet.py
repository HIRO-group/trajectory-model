# coding: utf-8
"""Command-line NatNet client application for testing.

Copyright (c) 2017, Matthew Edwards.  This file is subject to the 3-clause BSD
license, as found in the LICENSE file in the top-level directory of this
distribution and at https://github.com/mje-nz/python_natnet/blob/master/LICENSE.
No part of python_natnet, including this file, may be copied, modified,
propagated, or distributed except according to the terms contained in the
LICENSE file.
"""

from __future__ import print_function

import argparse
import time

import attr
import natnet
import csv
import atexit
import numpy as np

from constants import DATA_DIR, E_ID, TRAJECTORY_DURATION


csvfile = open(DATA_DIR, '+a', newline='\n')
writer = csv.writer(csvfile)

def exit_handler():
    print('Closing file...')
    csvfile.close()

atexit.register(exit_handler)

@attr.s
class ClientApp(object):

    _client = attr.ib()
    _quiet = attr.ib()
    _last_printed = attr.ib(0)

    @classmethod
    def connect(cls, server_name=None, quiet=False):
        client = natnet.Client.connect(server_name)
        if client is None:
            return None
        return cls(client, quiet)

    def run(self):
        self.start_time = time.time()
        self._client.set_callback(self.callback)
        self._client.spin()

    def callback(self, rigid_bodies, skeletons, markers, timing):
        experiment_id = E_ID
        if time.time() - self.start_time > TRAJECTORY_DURATION:
            print(f"Time has passed more than {TRAJECTORY_DURATION} seconds")
        elif rigid_bodies:
            for r in rigid_bodies:
                x, y, z = r.position
                a, b, c, d = r.orientation
                timestamp = timing.timestamp        
                print("Recieved: ", experiment_id, timestamp, x, y, z, a, b, c, d)
                writer.writerow([experiment_id, timestamp, x, y, z, a, b, c, d])

def main():
    try:
        time.sleep(3)
        print("starting now ...")
        app = ClientApp.connect()
        app.run()
    
    except natnet.DiscoveryError as e:
        print('Error:', e)


if __name__ == '__main__':
    main()