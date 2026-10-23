# psutil.Process

Version: 6.1.1

psutil.Process(pid=None)

Represents an OS process with the given PID.
If PID is omitted current process PID (os.getpid()) is used.
Raise NoSuchProcess if PID does not exist.

Note that most of the methods of this class do not make sure that
the PID of the process being queried has been reused. That means
that you may end up retrieving information for another process.

The only exceptions for which process identity is pre-emptively
checked and guaranteed are:

 - parent()
 - children()
 - nice() (set)
 - ionice() (set)
 - rlimit() (set)
 - cpu_affinity (set)
 - suspend()
 - resume()
 - send_signal()
 - terminate()
 - kill()

To prevent this problem for all other methods you can use
is_running() before querying the process.
