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
from datetime import datetime
import attr
import natnet
import csv
import atexit


DIR_PREFIX = '/home/ava/projects/trajectory-model/data/mocap_new_cups/'
file_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# tumbler/30  talltumbler/70
# talltumbler/50  talltumbler/80
# shorttumble/30  shorttumble/70
# wineglass/30 wineglass/70
# spilled spill-free

DIR_PATH = DIR_PREFIX + 'tumbler/30/spill-free/'

file_path = DIR_PATH + file_name + '.csv'
collected_data = []

def exit_handler():
    inp = int(input("save file? 1 or 0\n"))
    if inp == 1:
        with open(file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in collected_data:
                csv_writer.writerow(row)
        print("File saved to ", file_path)
    else:
        print("File not saved")

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
        if rigid_bodies:
            for r in rigid_bodies:
                x, y, z = r.position
                a, b, c, d = r.orientation
                timestamp = timing.timestamp
                print("r: ", r)
                print("recieving data (x, y, z, a, b, c, d): ", x, y, z, a, b, c, d)
                collected_data.append([timestamp, x, y, z, a, b, c, d])


def main():
    try:
        # time.sleep(1)
        print("Starting now ...")
        app = ClientApp.connect()
        app.run()
    
    except natnet.DiscoveryError as e:
        print('Error:', e)


if __name__ == '__main__':
    main()