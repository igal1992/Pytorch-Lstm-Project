import matplotlib.pyplot as plt
from ExcelClass import ExcelClass


def calculateAvg(firstNum, secondNum, _from, _to, arr):
    if abs(_to - _from <= 1):
        return

    pivot = int((_to + _from) / 2)  # pivot
    avgNum = (firstNum + secondNum) / 2  # not accurate in case that there is not half point
    arr[pivot] = avgNum  # update arr
    calculateAvg(firstNum, avgNum, _from, pivot, arr)
    calculateAvg(avgNum, secondNum, pivot, _to, arr)


def printGraph(original_data_arr, approximately_data_arr):
    x_axis = []
    for i in range(len(original_data_arr)):  # x axis will show iterations
        x_axis.append(i)

    # plotting the points
    plt.plot(x_axis, original_data_arr, label="original data")
    plt.plot(x_axis, approximately_data_arr, label="approximately data")

    # naming the x axis
    plt.xlabel('x - iterations')
    # naming the y axis
    plt.ylabel('y - Azimuth Value')

    # giving a title to my graph
    plt.title('Test')
    plt.savefig('graph.png')

    # function to show the plot
    plt.show()


# make an average function.
# ***important*** - for now we assume that starting index is 0 (the shadow zone begins at start)
def avgFunction(listOfFunctions, amountOfMissingData):
    arrToReturn = []  # arr that contains the avg values

    sumAtIndex = 0  # sum of specific index from all functions
    for i in range(amountOfMissingData):
        for j in range(len(listOfFunctions)):
            sumAtIndex += listOfFunctions[j][i]  # sums all values from all functions at the specific index
        arrToReturn.append(sumAtIndex / len(listOfFunctions))  # append an average
        sumAtIndex = 0  # reset

    return arrToReturn


# update function to be more accurate
# return new updated array
def updateFunction(predictionArr, mainSessionArr, disturbedUntil):
    valueAfterDisturbance = mainSessionArr[disturbedUntil + 1]  # get first value that we know in the arr
    lastValueOfPrediction = predictionArr[len(predictionArr) - 1]
    deltaY = abs(valueAfterDisturbance) - abs(lastValueOfPrediction)  # get the height difference
    for i in range(len(predictionArr)):
        if predictionArr[i] is not None:
            predictionArr[i] += deltaY if valueAfterDisturbance > lastValueOfPrediction else (deltaY * -1)  # update Y
    return predictionArr



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    xl_object = ExcelClass(2)
    list_of_days = []
    mainSession = xl_object.getDataFromColumn("elevation", 3)
    list_of_days.append(xl_object.getDataFromColumn("elevation", 598))
    list_of_days.append(xl_object.getDataFromColumn("elevation", 1188))

    x_axis_arr = []
    for i in range(len(mainSession)):  # x axis will show iterations
        x_axis_arr.append(i)

    # GRAPH DRAWING
    # plotting the points

    for i in range(len(x_axis_arr) - len(mainSession)):
        mainSession.append(None)
    plt.plot(x_axis_arr, mainSession, label="main")

    # for i in range(len(list_of_days)):
    #     if len(list_of_days[i]) < len(mainSession): # check if arrays different dimensions
    #         for j in range(len(x_axis_arr) - len(list_of_days[i])):
    #             list_of_days[i].append(None)
    #     plt.plot(x_axis_arr, list_of_days[i], label=("d"+str(i)))

    ourPredictArr = avgFunction(list_of_days, 500)
    ourPredictArr = updateFunction(ourPredictArr, mainSession, 500)
    for i in range(len(x_axis_arr) - len(ourPredictArr)):
        ourPredictArr.append(None)
    plt.plot(x_axis_arr, ourPredictArr, label="d4")

    # naming the x axis
    plt.xlabel('x - iterations')
    # naming the y axis
    plt.ylabel('y - rainLoss(dB)')

    # giving a title to my graph
    plt.title('Test')
    plt.savefig('graph.png')

    # function to show the plot
    plt.show()
