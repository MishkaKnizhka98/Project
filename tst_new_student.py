from my_functions import *

x, y = load_data("test_data/test_data.csv")
lorenzo = new_student(x)

print(lorenzo["school"] == "Moscow")