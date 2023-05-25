import natnet
client = natnet.Client.connect()
client.set_callback(
    lambda rigid_bodies, markers, timing: print(rigid_bodies))
client.spin()
