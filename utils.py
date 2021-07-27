import math
import random

from ExcelClass import ExcelClass

fName = "data23"

exlObject = ExcelClass(2, "data72")



# ----------------------------------calculating the disturbance angle-----------------------------------------#
def calculateAngle(buildingHeight, distanceFromBuilding, flag):
    elav = exlObject.elevationFix("data10")
    if flag:
        maxAngle = 175 - math.degrees(math.atan(buildingHeight / distanceFromBuilding))
    else:
        maxAngle = math.degrees(math.atan(buildingHeight / distanceFromBuilding))
    for i in range(len(elav)):
        if maxAngle < elav[i]:
            index = i
            break
    return index, maxAngle  # returns index and maximum angle of loss


# This method returns the  free space loss. (lf)
# d - distance col D
# f - frequency(GHz)
def calculateFreeSpaceLoss(d, f):
    return 32.45 + 20 * (math.log(d, 10) + math.log(f, 10))


# This method gets the parameters below and calculate&return the buildingLoss. (le)
# pt - transmitted power col F
# pr - received power
# gt - train gain
# gr - received gain
# lf - free space loss (calculated in other method)
def calculateBuildingLoss(pt, pr, gt, gr, lf):
    if (pt - pr) - lf + (gt + gr) < 0:
        return 0
    return (pt - pr) - lf + (gt + gr)


# calculates the array of losses based on the function in the excel from first index to last index
def calculateArrayLoss(firstIndex, lastIndex, index, ghz):
    pt = 30
    gr = exlObject.getDataFromColumn("rcvrGain", index)
    gt = exlObject.getDataFromColumn("XmtrGain", index)
    pr = exlObject.getDataFromColumn("rcvd_iso_power", index)
    d = exlObject.getDataFromColumn("range", index)
    lossArr = []
    for i in range(firstIndex, lastIndex):
        lf = calculateFreeSpaceLoss(d[i]*1000, ghz)
        lossArr.append(round(calculateBuildingLoss(pt, pr[i], gt[i], gr[i], lf), 2))
        print(round(calculateBuildingLoss(pt, pr[i], gt[i], gr[i], lf), 2))
    return lossArr


# normalize the columns given the col name and the index to start normalizing
def calculateSub(colName, index, ghz):
    array = []
    gt = exlObject.getDataFromColumn(colName, index)
    temp1 = gt[0] + ghz
    print(temp1)
    for i in range(int(len(gt) / 2)):
        temp = random.uniform(0, 0.2)
        temp1 = round((temp1 - temp), 4)
        array.append(temp1)
        print(array[i])
    for i in range(int(len(gt) / 2) - 1, len(gt)):
        temp = random.uniform(0, 0.2)
        temp1 = round((temp1 + temp), 4)
        array.append(temp1)
        print(array[i])
    exlObject.setDataInExcel(colName, array, fName, False)


if __name__ == '__main__':
    i = 0
    index = exlObject.setStartingRow("range", i)
    #calculateSub("XmtrGain", index, 72)
    #calculateSub("rcvrGain", index, 72)

    elav = exlObject.elevationFix("data72")
    angleIndex, _ = calculateAngle(40, 200, False)
    calculateArrayLoss(0, angleIndex - 1, index, 72)
    print("--------------------------------------------------------------------")
    angleIndex, _ = calculateAngle(40, 200, True)
    calculateArrayLoss(angleIndex, len(elav) - 1, index, 72)
