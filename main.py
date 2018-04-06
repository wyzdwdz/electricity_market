from pandas import DataFrame
import pandas as pd
from numpy import array
import numpy as np
import matplotlib.pyplot as plt


def data_maker(num_plants, num_users):
    """
    随机生成n个plants和m个users的报价投标数据，每家plants报3次价，每家users投2次标。
    数量在50MWh和200MWh之间（取50的整数倍)，价格在10美元/MWh和25美元/MWh之间。
    """
    com_plants = ['Plant' + str(i + 1) for i in range(num_plants)]
    com_plants_in_pd = []
    for i in com_plants:
        for j in range(3):
            com_plants_in_pd.append(i)

    com_users = ['User' + str(i + 1) for i in range(num_users)]
    com_users_in_pd = []
    for i in com_users:
        for j in range(2):
            com_users_in_pd.append(i)

    quan_plants = (np.random.randint(4, size=3 * num_plants) + 1) * 50
    quan_users = (np.random.randint(4, size=2 * num_users) + 1) * 50
    pri_plants = (np.random.randint(1000, size=3 * num_plants) + 1) * 0.01 + 15.
    pri_users = (np.random.randint(1000, size=2 * num_users) + 1) * 0.01 + 15.

    data_plants = {'Company': com_plants_in_pd, 'Quantity(WMh)': quan_plants, 'Price(US$/WMh)': pri_plants}
    frame_plants = DataFrame(data_plants, index=['Quotation' for i in range(3 * num_plants)])
    data_users = {'Company': com_users_in_pd, 'Quantity(WMh)': quan_users, 'Price(US$/WMh)': pri_users}
    frame_users = DataFrame(data_users, index=['Bidding' for i in range(2 * num_users)])
    frame = pd.concat([frame_plants, frame_users])
    frame = DataFrame(frame, columns=['Company', 'Quantity(WMh)', 'Price(US$/WMh)'])

    return frame_plants, frame_users, frame


def find_delivery_point(frame_plants, frame_users):
    """
    1. 生成两个array类型——plants_step、users_step，用以为作图标点。
    2. 寻找报价与投标两条曲线的交点，以确定交割价格。交割点保存为point或point1、point2。
    """
    frame_plants_sorted = frame_plants.sort_values(by='Price(US$/WMh)')
    frame_users_sorted = frame_users.sort_values(by='Price(US$/WMh)', ascending=False)

    plants_step = array(
        [list(frame_plants_sorted['Quantity(WMh)'].cumsum()), list(frame_plants_sorted['Price(US$/WMh)'])])
    users_step = array([frame_users_sorted['Quantity(WMh)'].cumsum(), frame_users_sorted['Price(US$/WMh)']])
    plants_insert = array([0, plants_step[1][0]])
    plants_step = np.insert(plants_step, 0, plants_insert, axis=1)
    users_insert = array([0, users_step[1][0]])
    users_step = np.insert(users_step, 0, users_insert, axis=1)

    plants_step_temp = array([[plants_step[0][0], plants_step[0][1]], [plants_step[1][0], plants_step[1][1]]])
    for i in range(plants_step[0].size - 2):
        if plants_step[1][i + 2] != plants_step[1][i + 1]:
            plants_step_temp = np.append(plants_step_temp,
                                         array([array([plants_step[0][i + 2], plants_step[1][i + 2]])]).T, axis=1)
    plants_step_temp = np.append(plants_step_temp,
                                 array([array([plants_step[0][-1], plants_step[1][-1]])]).T, axis=1)
    plants_step = plants_step_temp
    users_step_temp = array([[users_step[0][0], users_step[0][1]], [users_step[1][0], users_step[1][1]]])
    for i in range(users_step[0].size - 2):
        if users_step[1][i + 2] != users_step[1][i + 1]:
            users_step_temp = np.append(users_step_temp,
                                        array([array([users_step[0][i + 2], users_step[1][i + 2]])]).T, axis=1)
    users_step_temp = np.append(users_step_temp,
                                array([array([users_step[0][-1], users_step[1][-1]])]).T, axis=1)
    users_step = users_step_temp

    plants_points = array([[plants_step[0][0]], [plants_step[1][0]]])
    plants_iteration = np.delete(plants_step, 0, axis=1)
    plants_iteration = np.delete(plants_iteration, -1, axis=1)
    i = 1
    for temp in plants_iteration.T:
        plants_points = np.insert(plants_points, plants_points[0].size, temp, axis=1)
        plants_points = np.insert(plants_points, plants_points[0].size,
                                  array([temp[0], plants_step[1][i + 1]]), axis=1)
        i = i + 1
    plants_points = np.concatenate((plants_points, [[plants_step[0][-1]], [plants_step[1][-1]]]), axis=1)  
                                                                        #标注出梯形图上每一个折点的坐标plants_points，users_points同理

    users_points = array([[users_step[0][0]], [users_step[1][0]]])
    users_iteration = np.delete(users_step, 0, axis=1)
    users_iteration = np.delete(users_iteration, -1, axis=1)
    i = 1
    for temp in users_iteration.T:
        users_points = np.insert(users_points, users_points[0].size, temp, axis=1)
        users_points = np.insert(users_points, users_points[0].size, array([temp[0], users_step[1][i + 1]]), axis=1)
        i = i + 1
    users_points = np.concatenate((users_points, [[users_step[0][-1]], [users_step[1][-1]]]), axis=1)

    flag = 0      #flag为交点类型标志
    for i in range(int(plants_points[0].size / 2 - 1)):           #寻找users阶梯线水平交于plants阶梯线的情况，交点只有1个，若存在，置flag=1
        index = np.where(
            np.all([users_points[1] > plants_points[1][2 * i + 1], users_points[1] < plants_points[1][2 * i + 2]],
                   axis=0))[0]
        for j in range(index.size - 1):
            if users_points[0][index[j]] < plants_points[0][2 * i + 1] and users_points[0][index[j] + 1] > \
                    plants_points[0][2 * i + 2]:
                point = [plants_points[0][2 * i + 1], users_points[1][index[j]]]
                flag = 1
                return [plants_step, users_step, flag, point]

    for i in range(int(plants_points[0].size / 2)):               #寻找users阶梯线竖直交于plants阶梯线的情况，交点只有1个，若存在，仍置flag=1
        index = np.where(
            np.all([users_points[0] > plants_points[0][2 * i], users_points[0] < plants_points[0][2 * i + 1]], axis=0))[
            0]
        for j in range(index.size - 1):
            if users_points[1][index[j]] > plants_points[1][2 * i] and users_points[1][index[j] + 1] < plants_points[1][
                                2 * i + 1]:
                point = [users_points[0][index[j]], plants_points[1][2 * i]]
                flag = 1
                return [plants_step, users_step, flag, point]

    for i in range(int(plants_points[0].size / 2 - 1)):     
                                      #寻找users阶梯线竖直交于plants阶梯线的情况，但交点有多个，此时交割价格根据报价先后确定，若存在，置flag=2
        index = np.where(users_points[0] == plants_points[0][2 * i + 1])[0]
        if index.size == 1:
            break
        else:
            for j in range(int(index.size / 2)):
                if sorted([plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                           users_points[1][index[j + 1]]], reverse=True) == [users_points[1][index[j]],
                                                                             plants_points[1][2 * i + 2],
                                                                             plants_points[1][2 * i + 1],
                                                                             users_points[1][
                                                                                 index[j + 1]]] and np.unique(
                    [plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                     users_points[1][index[j + 1]]]).size == 4:
                    point1 = [plants_points[0][2 * i + 2], plants_points[1][2 * i + 2]]
                    point2 = [plants_points[0][2 * i + 1], plants_points[1][2 * i + 1]]
                    flag = 2
                    return [plants_step, users_step, flag, point1, point2]
                elif sorted([plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                             users_points[1][index[j + 1]]], reverse=True) == [users_points[1][index[j]],
                                                                               plants_points[1][2 * i + 2],
                                                                               users_points[1][index[j + 1]],
                                                                               plants_points[1][
                                                                                                   2 * i + 1]] and np.unique(
                    [plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                     users_points[1][index[j + 1]]]).size == 4:
                    point1 = [plants_points[0][2 * i + 2], plants_points[1][2 * i + 2]]
                    point2 = [users_points[0][index[j + 1]], users_points[1][index[j + 1]]]
                    flag = 2
                    return [plants_step, users_step, flag, point1, point2]
                elif sorted([plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                             users_points[1][index[j + 1]]], reverse=True) == [plants_points[1][2 * i + 2],
                                                                               users_points[1][index[j]],
                                                                               plants_points[1][2 * i + 1],
                                                                               users_points[1][
                                                                                   index[j + 1]]] and np.unique(
                    [plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                     users_points[1][index[j + 1]]]).size == 4:
                    point1 = [users_points[0][index[j]], users_points[1][index[j]]]
                    point2 = [plants_points[0][2 * i + 1], plants_points[1][2 * i + 1]]
                    flag = 2
                    return [plants_step, users_step, flag, point1, point2]
                elif sorted([plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                             users_points[1][index[j + 1]]], reverse=True) == [plants_points[1][2 * i + 2],
                                                                               users_points[1][index[j]],
                                                                               users_points[1][index[j + 1]],
                                                                               plants_points[1][
                                                                                                   2 * i + 1]] and np.unique(
                    [plants_points[1][2 * i + 1], plants_points[1][2 * i + 2], users_points[1][index[j]],
                     users_points[1][index[j + 1]]]).size == 4:
                    point1 = [users_points[0][index[j]], users_points[1][index[j]]]
                    point2 = [users_points[0][index[j + 1]], users_points[1][index[j + 1]]]
                    flag = 2
                    return [plants_step, users_step, flag, point1, point2]

    for i in range(int(plants_points[0].size / 2)):  #寻找users阶梯线水平交于plants阶梯线的情况，交点有多个，但交割价格确定，若存在，置flag=3
        index = np.where(users_points[1] == plants_points[1][2 * i])[0]
        if index.size == 1:
            break
        else:
            for j in range(int(index.size / 2)):
                if sorted([plants_points[0][2 * i], plants_points[0][2 * i + 1], users_points[0][index[j]],
                           users_points[0][index[j + 1]]], reverse=True) == [users_points[0][index[j + 1]],
                                                                             plants_points[0][2 * i + 1],
                                                                             plants_points[0][2 * i],
                                                                             users_points[0][index[j]]]:
                    point1 = [plants_points[0][2 * i + 1], plants_points[1][2 * i + 1]]
                    point2 = [plants_points[0][2 * i], plants_points[1][2 * i]]
                    flag = 3
                    return [plants_step, users_step, flag, point1, point2]
                elif sorted([plants_points[0][2 * i], plants_points[0][2 * i + 1], users_points[0][index[j]],
                             users_points[0][index[j + 1]]], reverse=True) == [users_points[0][index[j + 1]],
                                                                               plants_points[0][2 * i + 1],
                                                                               users_points[0][index[j]],
                                                                               plants_points[0][2 * i]]:
                    point1 = [plants_points[0][2 * i + 1], plants_points[1][2 * i + 1]]
                    point2 = [users_points[0][index[j]], users_points[1][index[j]]]
                    flag = 3
                    return [plants_step, users_step, flag, point1, point2]
                elif sorted([plants_points[0][2 * i], plants_points[0][2 * i + 1], users_points[0][index[j]],
                             users_points[0][index[j + 1]]], reverse=True) == [plants_points[0][2 * i + 1],
                                                                               users_points[0][index[j + 1]],
                                                                               plants_points[0][2 * i],
                                                                               users_points[0][index[j]]]:
                    point1 = [users_points[0][index[j + 1]], users_points[1][index[j + 1]]]
                    point2 = [plants_points[0][2 * i], plants_points[1][2 * i]]
                    flag = 3
                    return [plants_step, users_step, flag, point1, point2]
                elif sorted([plants_points[0][2 * i], plants_points[0][2 * i + 1], users_points[0][index[j]],
                             users_points[0][index[j + 1]]], reverse=True) == [plants_points[0][2 * i + 1],
                                                                               users_points[0][index[j + 1]],
                                                                               users_points[0][index[j]],
                                                                               plants_points[0][2 * i]]:
                    point1 = [users_points[0][index[j + 1]], users_points[1][index[j + 1]]]
                    point2 = [users_points[0][index[j]], users_points[1][index[j]]]
                    flag = 3
                    return [plants_step, users_step, flag, point1, point2]

    if flag == 0:          #无交割情况
        return [plants_step, users_step, flag]


def drawing(delivery_point_pack):
    """
    作阶梯图并在图中标出交割点
    """
    flag = delivery_point_pack[2]
    if flag == 0:
        plants_step = delivery_point_pack[0]
        users_step = delivery_point_pack[1]
    elif flag == 1:
        plants_step = delivery_point_pack[0]
        users_step = delivery_point_pack[1]
        point = delivery_point_pack[3]
    elif flag == 2 or flag == 3:
        plants_step = delivery_point_pack[0]
        users_step = delivery_point_pack[1]
        point1 = delivery_point_pack[3]
        point2 = delivery_point_pack[4]

    if flag == 0:
        print("两曲线无交割点，故无法定价")
    elif flag == 1:
        print("两曲线有一个交割点，定价为" + str(delivery_point_pack[3][1]) + "US$/MWh")
    elif flag == 2:
        print(
            "两曲线多个交割点，定价区间为" + str(delivery_point_pack[3][1]) + "US$/MWh和" + str(delivery_point_pack[4][1]) + "US$/MWh")
    elif flag == 3:
        print("两曲线有多个交割点，定价为" + str(delivery_point_pack[3][1]) + "US$/MWh")

    plt.figure()
    plt.step(plants_step[0], plants_step[1])
    plt.step(users_step[0], users_step[1])

    if flag == 0:
        plt.show()
    elif flag == 1:
        plt.scatter(point[0], point[1])
        plt.show()
    elif flag == 2 or flag == 3:
        plt.scatter(point1[0], point1[1])
        plt.scatter(point2[0], point2[1])
        plt.show()


if __name__ == '__main__':
    """
    主函数程序，提供用户交互功能
    """
    print("共有多少家企业参与报价？（请输入整数）：", end='')
    num_plants = int(input())
    print("共有多少家企业参与投标？（请输入整数）：", end='')
    num_users = int(input())
    frame_plants, frame_users, frame = data_maker(num_plants, num_users)
    print("\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        print(frame)                                              #打印完整的报价投标表
    print("\n")
    delivery_point_pack = find_delivery_point(frame_plants, frame_users)
    drawing(delivery_point_pack)