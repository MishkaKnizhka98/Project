from my_functions import *

pd.set_option('mode.chained_assignment', None)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
x, y = load_data("data/student-mat.csv")

x_dummy = dummy_matrices(x)

w, b, r_sq = compute_model(x_dummy, y)

print("Predicted w: ", w, "\n", "Predicted b: ", "%.3f" % b, "\n",
      "Coefficient of determination: ", "%.3f" % r_sq)

lorenzo = new_student(x)

lorenzo_dummy = dummy_matrix_of_new_student(lorenzo,x)


print("The predicted G3 for Lorenzo is : ", "%.3f" % predict(lorenzo_dummy, w, b))

plot(x,y,w,b)