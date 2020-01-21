import signal
import time
import sys
from functools import partial


def terminate_search(thread, sig, frame):
    print("Terminating search process...")
    thread.set()


def confirm_exit(thread, sig, frame):
    print("\nTo terminate search process, press Ctrl-C again. Otherwise, resuming in 5 seconds.")
    signal.signal(signal.SIGINT, partial(terminate_search, thread))
    time.sleep(5)
    print("Resuming search process...")
    signal.signal(signal.SIGINT, partial(confirm_exit, thread))


def setup_signal(thread):
    signal.signal(signal.SIGINT, partial(confirm_exit, thread))


def remove_signal():
    signal.signal(signal.SIGINT, signal.default_int_handler)
