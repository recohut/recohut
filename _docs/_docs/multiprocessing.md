---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="VAGsPXWlc_M8" -->
# Python Multiprocessing and Multithreading
<!-- #endregion -->

<!-- #region id="DcziXhSDHNf0" -->
## Threads
<!-- #endregion -->

```python id="Ah9yKvPDtYB1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1634789764620, "user_tz": -330, "elapsed": 2226, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="54d420c6-19c4-42cc-d4f5-300ff86cd84c"
from threading import current_thread, Thread
from time import sleep


def print_hello():
    sleep(2)
    print("{}: Hello".format(current_thread().name))


def print_message(msg):
    sleep(1)
    print("{}: {}".format(current_thread().name, msg))


# creating threads
t1 = Thread(target=print_hello, name="Th 1")
t2 = Thread(target=print_hello, name="Th 2")
t3 = Thread(target=print_message, args=["Good morning"], name="Th 3")

# start the threads
t1.start()
t2.start()
t3.start()

"""
After creating an object of the Thread class, we need to start the thread by 
using the start method. To make the main program or thread wait until the newly 
created thread object(s) finishes, we need to use the join method. The join 
method makes sure that the main thread (a calling thread) waits until the thread 
on which the join method is called completes its execution.
"""

# wait till all are done
t1.join()
t2.join()
t3.join()
```

<!-- #region id="DO8658tHDjPB" -->
Thread 1 and thread 2 have more sleep time than thread 3, so thread 3 will always finish first. Thread 1 and thread 2 can finish in any order depending on who gets hold of the processor first.

In this program, we implemented the following:

- We created two simple functions, print_hello and print_message, that are to be used by the threads. We used the sleep function from the time module in both functions to make sure that the two functions finish their execution time at different times.
- We created three Thread objects. Two of the three objects will execute one function (print_hello) to illustrate the code sharing by the threads, and the third thread object will use the second function (print_message), which takes one argument as well.
- We started all three threads one by one using the start method.
We waited for each thread to finish by using the join method.
<!-- #endregion -->

<!-- #region id="Nl8SXImoHQHC" -->
## Deamon threads
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kmde8zioE0b9" executionInfo={"status": "ok", "timestamp": 1634790568169, "user_tz": -330, "elapsed": 1361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0eae820b-2d1e-4fa9-85ea-1b3c8e3f7243"
from threading import current_thread, Thread
from time import sleep

def daeom_func():
    #print(threading.current_thread().isDaemon())
    sleep(3)
    print("{}: Hello from daemon".format
          (current_thread().name))


def nondaeom_func():
    #print(threading.current_thread().isDaemon())
    sleep(1)
    print("{}: Hello from non-daemon".format(
        current_thread().name))


# creating threads
t1 = Thread(target=daeom_func, name="Daemon Thread",daemon=True)
t2 = Thread(target=nondaeom_func, name="Non-Daemon Thread")

# start the threads
t1.start()
t2.start()

t2.join()


print("Exiting the main program")
```

<!-- #region id="7GjQSuzaFR5I" -->
In this code example, we created one daemon and one non-daemon thread. The daemon thread (daeom_func) is executing a function that has a sleep time of 3 seconds, whereas the non-daemon thread is executing a function (nondaeom_func) that has a sleep time of 1 second. The sleep time of the two functions is set to make sure the non-daemon thread finishes its execution first.

Since we did not use a join method in any thread, the main thread exits first, and then the non-daemon thread finishes a bit later with a print message. But there is no print message from the daemon thread. This is because the daemon thread is terminated as soon as the non-daemon thread finishes its execution. If we change the sleep time in the nondaeom_func function to 5, the console output for deamon thread will also be printed.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xWOd2rmnFxhT" executionInfo={"status": "ok", "timestamp": 1634790579972, "user_tz": -330, "elapsed": 5563, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c3d125d4-1a73-44cd-b37c-9d780db9c0c7"
from threading import current_thread, Thread
from time import sleep

def daeom_func():
    #print(threading.current_thread().isDaemon())
    sleep(3)
    print("{}: Hello from daemon".format
          (current_thread().name))


def nondaeom_func():
    #print(threading.current_thread().isDaemon())
    sleep(5)
    print("{}: Hello from non-daemon".format(
        current_thread().name))


# creating threads
t1 = Thread(target=daeom_func, name="Daemon Thread",daemon=True)
t2 = Thread(target=nondaeom_func, name="Non-Daemon Thread")

# start the threads
t1.start()
t2.start()

t2.join()


print("Exiting the main program")
```

<!-- #region id="segpRq8-Fw5J" -->
## Synchronizing threads
<!-- #endregion -->

<!-- #region id="dwNCfSA0HU25" -->
Multiple threads accessing the critical section at the same time may try to access or change the data at the same time, which may result in unpredictable results on the data. This situation is called a race condition.

To illustrate the concept of the race condition, we will implement a simple program with two threads, and each thread increments a shared variable 1 million times. We chose a high number for the increment to make sure that we can observe the outcome of the race condition. The race condition may also be observed by using a lower value for the increment cycle on a slower CPU. In this program, we will create two threads that are using the same function (inc in this case) as the target. The code for accessing the shared variable and incrementing it by 1 occurs in the critical section, and the two threads are accessing it without any protection.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vj1wzeW3H_kc" executionInfo={"status": "ok", "timestamp": 1634790946531, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a955abb-f1bf-4ec8-a3c3-45774eb10b03"
from threading import Thread

def inc():
    global x
    for _ in range(1000000):
        x+=1

#global variable
x = 0

# creating threads
t1 = Thread(target=inc, name="Th 1")
t2 = Thread(target=inc, name="Th 2")

# start the threads
t1.start()
t2.start()

#wait for the threads
t1.join()
t2.join()

print("final value of x :", x)
```

<!-- #region id="Oh9a0tN2IAgU" -->

The expected value of x at the end of the execution is 2,000,000, which will not be observed in the console output. Every time we execute this program, we will get a different value of x that's a lot lower than 2,000,000. This is because of the race condition between the two threads. Let's look at a scenario where threads Th 1 and Th 2 are running the critical section (x+=1) at the same time. Both threads will ask for the current value of x. If we assume the current value of x is 100, both threads will read it as 100 and increment it to a new value of 101. The two threads will write back to the memory the new value of 101. This is a one-time increment and, in reality, the two threads should increment the variable independently of each other and the final value of x should be 102. How can we achieve this? This is where thread synchronization comes to the rescue.

Thread synchronization can be achieved by using a Lock class from the threading module. The lock is implemented using a semaphore object provided by the operating system. A semaphore is a synchronization object at the operating system level to control access to the resources and data for multiple processors and threads. The Lock class provides two methods, acquire and release, which are described next:

The acquire method is used to acquire a lock. A lock can be blocking (default) or non-blocking. In the case of a blocking lock, the requesting thread's execution is blocked until the lock is released by the current acquiring thread. Once the lock is released by the current acquiring thread (unlocked), then the lock is provided to the requesting thread to proceed. In the case of a non-blocking acquire request, the thread execution is not blocked. If the lock is available (unlocked), then the lock is provided (and locked) to the requesting thread to proceed, otherwise the requesting thread gets False as a response.

The release method is used to release a lock, which means it resets the lock to an unlocked state. If there is any thread blocking and waiting for the lock, it will allow one of the threads to proceed.

The above code example is revised with the use of a lock around the increment statement on the shared variable x. In this revised example, we created a lock at the main thread level and then passed it to the inc function to acquire and release a lock around the shared variable. The complete revised code example is as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="z9C7c6aQIfVn" executionInfo={"status": "ok", "timestamp": 1634790976583, "user_tz": -330, "elapsed": 3368, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="80668156-a9b0-449f-c3de-64f6ed802172"
from threading import Lock, Thread

def inc_with_lock (lock):
    global x
    for _ in range(1000000):
        lock.acquire()
        x+=1
        lock.release()

x = 0
mylock = Lock()
# creating threads
t1 = Thread(target=inc_with_lock , args=(mylock,), name="Th 1")
t2 = Thread(target=inc_with_lock , args=(mylock,), name="Th 2")

# start the threads
t1.start()
t2.start()

#wait for the threads
t1.join()
t2.join()
print("final value of x :", x)
```

<!-- #region id="DeDa0wq1IpvX" -->
However, locks have to be used carefully because improper use of locks can result in a deadlock situation. Suppose a thread acquires a lock on resource A and is waiting to acquire a lock on resource B. But another thread already holds a lock on resource B and is looking to acquire a lock resource A. The two threads will wait for each other to release the locks, but it will never happen. To avoid deadlock situations, the multithreading and multiprocessing libraries come with mechanisms such as adding a timeout for a resource to hold a lock, or using a context manager to acquire locks.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KpsW0DOIKlfN" executionInfo={"status": "ok", "timestamp": 1634791500671, "user_tz": -330, "elapsed": 2578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="90e85790-7ff1-4c5f-c176-4033717b4d79"
from queue import Queue
from threading import Thread
from time import sleep

"""
To implement a custom thread class, we need to override the init and run methods. 
In the init method, it is required to call the init method of the superclass 
(the Thread class). The run method is the execution part of the thread.
"""

class MyWorker(Thread):
   def __init__(self, name, q):
      Thread.__init__(self)
      self.name = name
      self.queue = q

   def run(self):
      while True:
          item = self.queue.get()
          sleep(1)
          try:
              print ("{}: {}".format(self.name, item))
          finally:
            self.queue.task_done()

#filling the queue
myqueue = Queue()
for i in range(10):
    myqueue.put("Task {}".format(i+1))

# creating threads
for i in range(5):
    worker = MyWorker("Th {}".format(i+1), myqueue)
    worker.daemon = True
    worker.start()

myqueue.join()
```

<!-- #region id="kgH2QTDWKp6P" -->
In this code example, we created five worker threads using the custom thread class (MyThread). These five worker threads access the queue to get the task item from it. After getting the task item, the threads sleep for 1 second and then print the thread name and the task name. For each get call for an item of a queue, a subsequent call of task_done() indicates that the processing of the task has been completed.

It is important to note that we used the join method on the myqueue object and not on the threads. The join method on the queue blocks the main thread until all items in the queue have been processed and completed (task_done is called for them). This is a recommended way to block the main thread when a queue object is used to hold the tasks' data for threads.
<!-- #endregion -->

<!-- #region id="ERAFk81CNibG" -->
## Multiprocessing
<!-- #endregion -->

<!-- #region id="NByiNzuMNsh9" -->
For multiprocessing programming, Python provides a multiprocessing package that is very similar to the multithreading package. The multiprocessing package includes two approaches to implement multiprocessing, which are using the Process object and the Pool object.
<!-- #endregion -->

<!-- #region id="pMYmJbMKOkP1" -->
### Using the Process object
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="31R3aiU-NjlI" executionInfo={"status": "ok", "timestamp": 1634792480465, "user_tz": -330, "elapsed": 2886, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="805c3b9e-0b28-4ff9-f204-97f32e227b2e"
import os
from multiprocessing import Process, current_process as cp
from time import sleep


def print_hello():
    sleep(2)
    print("{}-{}: Hello".format(os.getpid(), cp().name))


def print_message(msg):
    sleep(1)
    print("{}-{}: {}".format(os.getpid(), cp().name, msg))


def main():
    processes = []

    # creating process
    processes.append(Process(target=print_hello, name="Process 1"))
    processes.append(Process(target=print_hello, name="Process 2"))
    processes.append(Process(target=print_message,
                             args=["Good morning"], name="Process 3"))

    # start the process
    for p in processes:
        p.start()

    # wait till all are done
    for p in processes:
        p.join()

    print("Exiting the main process")

if __name__ == '__main__':
    main()
```

<!-- #region id="plvfczEWOZCQ" -->
As already mentioned, the methods used for the Process object are pretty much the same as those used for the Thread object.
<!-- #endregion -->

<!-- #region id="_YqtcLuaOloW" -->
### Using the Pool object
<!-- #endregion -->

<!-- #region id="9alaNlgROm7r" -->
The Pool object offers a convenient way (using its map method) of creating processes, assigning functions to each new process, and distributing input parameters across the processes. We selected the code example with a pool size of 3 but provided input parameters for five processes. The reason for setting the pool size to 3 is to make sure a maximum of three child processes are active at a time, regardless of how many parameters we pass with the map method of the Pool object. The additional parameters will be handed over to the same child processes as soon they finish their current execution.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XTdU0ZuDOzTl" executionInfo={"status": "ok", "timestamp": 1634792656580, "user_tz": -330, "elapsed": 2518, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d1e3298f-cec8-4e78-9b06-78eedb88d3c7"
import os
from multiprocessing import Process, Pool, current_process as cp
from time import sleep


def print_message(msg):
    sleep(1)
    print("{}-{}: {}".format(os.getpid(), cp().name, msg))


def main():
    # creating process from a pool
    with Pool(3) as proc:
        proc.map(print_message, ["Orange", "Apple", "Banana",
                                 "Grapes","Pears"])

    print("Exiting the main process")

if __name__ == '__main__':
    main()
```

<!-- #region id="xrxKmLRDPEIi" -->
The magic of distributing input parameters to a function that is tied to a set of pool processes is done by the map method. The map method waits until all functions complete their execution, and that is why there is no need to use a join method if the processes are created using the Pool object.

A few differences between using the Process object versus the Pool object are shown in the following table:

| Using the Pool object | Using the Process object |
| --------------------- | ------------------------ |
| Only active processes active in memory | All created processes stay in memory |
| Works better for large datasets and for repetitive tasks | Works better for small datasets |
| Processes block on I/O operation until I/O resource is granted | Processes are not blocked on the I/O operation |
<!-- #endregion -->

<!-- #region id="qesrcOjNQ_qo" -->
## Using asynchronous programming for responsive systems

The asyncio module is rich with features and supports creating and running Python coroutines, performing network I/O operations, distributing tasks to queues, and synchronizing concurrent code.

Coroutines are the functions that are to be executed asynchronously. A simple example of sending a string to the console output using a coroutine is as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 146} id="liAQy6s2RbQ-" executionInfo={"status": "error", "timestamp": 1634793492315, "user_tz": -330, "elapsed": 482, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ba12d93e-70c1-45b4-8ed1-038251aed1a8"
import asyncio
import time


async def say(delay, msg):
    await asyncio.sleep(delay)
    print(msg)

print("Started at ", time.strftime("%X"))

await say(1,"Good")
# asyncio.run(say(1,"Good"))
# asyncio.run(say(2, "Morning"))
print("Stopped at ", time.strftime("%X"))
```

```python id="_684kaKLRfMv"

```
