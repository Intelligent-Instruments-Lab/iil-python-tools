from threading import RLock, Lock

# _lock = Lock()

# using a reentrant lock should allow the user to lock around something
# which may be called from a @repeat, MIDI handler, etc without stalling
_lock = RLock()
