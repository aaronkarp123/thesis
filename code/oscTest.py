import argparse
import math

from pythonosc import dispatcher
from pythonosc import osc_server
import asyncio

def receive_message(unused_addr, args, audio):
    audio = "{1}".format(args[0], audio)
    print(audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port",
      type=int, default=12000, help="The port to listen on")
    args = parser.parse_args()

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/onoff", receive_message, "Audio")

    oscserverloop = asyncio.get_event_loop()
    server = osc_server.AsyncIOOSCUDPServer(
      (args.ip, args.port), dispatcher, oscserverloop)

    server.serve()
    while True:
      oscserverloop.stop()
      oscserverloop.run_forever()