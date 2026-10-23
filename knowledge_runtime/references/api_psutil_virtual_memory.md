# psutil.virtual_memory

Version: 6.1.1

psutil.virtual_memory()

Return statistics about system memory usage as a namedtuple
including the following fields, expressed in bytes:

 - total:
   total physical memory available.

 - available:
   the memory that can be given instantly to processes without the
   system going into swap.
   This is calculated by summing different memory values depending
   on the platform and it is supposed to be used to monitor actual
   memory usage in a cross platform fashion.

 - percent:
   the percentage usage calculated as (total - available) / total * 100

 - used:
    memory used, calculated differently depending on the platform and
    designed for informational purposes only:
    macOS: active + wired
    BSD: active + wired + cached
    Linux: total - free

 - free:
   memory not being used at all (zeroed) that is readily available;
   note that this doesn't reflect the actual memory available
   (use 'available' instead)

Platform-specific fields:

 - active (UNIX):
   memory currently in use or very recently used, and so it is in RAM.

 - inactive (UNIX):
   memory that is marked as not used.

 - buffers (BSD, Linux):
   cache for things like file system metadata.

 - cached (BSD, macOS):
   cache for various things.

 - wired (macOS, BSD):
   memory that is marked to always stay in RAM. It is never moved to disk.

 - shared (BSD):
   memory that may be simultaneously accessed by multiple processes.

The sum of 'used' and 'available' does not necessarily equal total.
On Windows 'available' and 'free' are the same.
