
import sys
import numpy as np

import MyModule as test

print("Module Imported")


dut = test.hello(5) 
print("Init value is:", dut.get_value())

dut.set_value(7)
print("Set value is:", dut.get_value())


golden = np.random.randint(-256, 255, (4,4,2,3))

# result = dut.copy(golden)

# print("Input array is:", golden)
# print("Copied array is:", result)

# if not np.array_equal(golden, result):
#     sys.exit('MISMATCH!')

print("Input array is:", golden)

res_a = dut.copy(golden[:,0:1,:,:])
print("Copied array A is:", res_a)

if not np.array_equal(golden[:,0:1,:,:], res_a):
    sys.exit('MISMATCH!')

res_b = dut.copy(golden[:,2:3,:,:])
print("Copied array B is:", res_b)

if not np.array_equal(golden[:,2:3,:,:], res_b):
    sys.exit('MISMATCH!')


print("Test Performed")
