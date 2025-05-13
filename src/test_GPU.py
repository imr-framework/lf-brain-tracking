import GPUtil
import tensorflow as tf
import os

print(tf.config.list_physical_devices('GPU'))

num_gpus = len(tf.config.list_physical_devices('GPU'))
print("Number of GPUs:", num_gpus)

# Rest of your code
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print("GPU Name:", gpu.name)
    print("GPU Memory:", gpu.memoryTotal)
    print("GPU Memory Used:", gpu.memoryUsed)
    print("GPU Memory Free:", gpu.memoryFree)
    print("GPU UUID:", gpu.uuid)
    print("GPU Load:", gpu.load)
    print("GPU Temperature:", gpu.temperature)
    print("GPU Driver Version:", gpu.driver)


# gpu = 'all'
# if gpu == 'all':
#     # Stay stuck in this loop until there is some gpu available with at least half capacity
#     gpus = []
#     while not gpus:
#         gpus = GPUtil.getAvailable(order='memory')
#         print(gpus)s