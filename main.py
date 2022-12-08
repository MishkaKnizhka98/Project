from my_functions import *
pd.set_option('mode.chained_assignment', None)  # To turn off SettingWithCopyWarning
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})  # To print the predicted w to 3 decimals places

# In order to train the model to predict final grade in Math, student-mat.csv is loaded
x, y = load_data("data/student-mat.csv")

# Training set x is exposed to dummy_matrices() in order to convert all categorical features with dummy matrices
x_dummy = dummy_matrices(x)

w, b, r_sq = compute_model(x_dummy, y)

# Printing predicted w, b and r_sq to 3 decimal places
print("Predicted w: ", w, "\n", "Predicted b: ", "%.3f" % b, "\n",
      "Coefficient of determination: ", "%.3f" % r_sq)

# Here we want to predict the final grade "G3" of a new student Lorenzo
# In order to turn Lorenzo's categorical features into dummy matrices, dummy_matrix_of_new_student() is used
lorenzo = new_student(x)

lorenzo_dummy = dummy_matrix_of_new_student(lorenzo,x)


print("The predicted G3 for Lorenzo is : ", "%.3f" % predict(lorenzo_dummy, w, b))

# In order to visually verify the accuracy of the trained model, we use plot()
# to depict dependencies of the output "G3" on numeric features
plot(x, y, w, b)