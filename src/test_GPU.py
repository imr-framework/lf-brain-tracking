import GPUtil
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

# gpu = 'all'
# if gpu == 'all':
#     # Stay stuck in this loop until there is some gpu available with at least half capacity
#     gpus = []
#     while not gpus:
#         gpus = GPUtil.getAvailable(order='memory')
#         print(gpus)