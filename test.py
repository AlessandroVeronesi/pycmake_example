
import MyModule as test

print("Module Imported")

a = test.hello(5) 
print("Value is:", a.get_value())

a.set_value(7)
print("Value is:", a.get_value())

print("Test Performed")
