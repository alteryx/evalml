import signal
import time
from functools import partial


def terminate_search(search_thread, exit_event, sig, frame):
    # print("Terminating search process...")
    # search_thread.exit()
    exit_event.set()
    remove_signal()


def confirm_exit(search_thread, exit_event, sig, frame):
    print("\nTo terminate search process, press Ctrl-C again. Otherwise, resuming in 5 seconds.")
    signal.signal(signal.SIGINT, partial(terminate_search, search_thread, exit_event))
    time.sleep(5)
    if not exit_event.is_set():
        print("\nResuming search process...")
        signal.signal(signal.SIGINT, partial(confirm_exit, search_thread, exit_event))


def setup_signal(search_thread, exit_event):
    signal.signal(signal.SIGINT, partial(confirm_exit, search_thread, exit_event))


def remove_signal():
    signal.signal(signal.SIGINT, signal.default_int_handler)
