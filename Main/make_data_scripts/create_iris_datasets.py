

with open('iris_data/iris.data', 'r') as iris_file:
    iris_data = list(iris_file.readlines())
    setosa = iris_data[: 50]
    versicolor = iris_data[50: 100]
    virginica = iris_data[100: -1]      

    with open('iris_data/setosa-versicolor.csv', 'w') as new_file:
        new_file.writelines(setosa)
        new_file.writelines(versicolor)
    
    with open('iris_data/versicolor-virginica.csv', 'w') as new_file:
        new_file.writelines(versicolor)
        new_file.writelines(virginica)

    with open('iris_data/virginica-setosa.csv', 'w') as new_file:
        new_file.writelines(virginica)
        new_file.writelines(setosa)