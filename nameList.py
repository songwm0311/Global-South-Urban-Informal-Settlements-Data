import scipy.io as sio
from select_data import Days
import calendar

from os.path import join

def name_list(year,month):

    list = []
    for step,i in enumerate(year):
        for k in range(len(month[step])):

            star = Days(i,month[step][k],1)

            monthRange = calendar.monthrange(i, month[step][k])[-1]
            # print(monthRange)
            for j in range(star,star+monthRange):
                for Ti in range(24):
                    list.append(str(i)+str(j).zfill(3)+ '-' + str(Ti).zfill(2) +'.mat')
    return  list


def PickListform(dir,train_name):
    JinList = sio.loadmat(dir)
    JinList = (JinList[list(JinList.keys())[3]]).tolist()

    s = []

    for item in JinList:
        for i in item:
            s.append(str(i))
    set_c = list(set(train_name) & set(s))

    return set_c

def FormTrainremovdList(trainlist,list1):
    trainlist = [i for i in trainlist if i not in list1]
    return trainlist


def get_intersection(BENDIlist, Genlist):
    """
    train_name_list    : [/home/jinxiangze/jinxiangze/dataPRE/CLDAS/pre/pcp-2020366.mat,...]
    rainstorm_name_list: [pcp-2017202-6,...]
    """
    list1 = []


    for x in BENDIlist:
        list1.append(x[6:])  # pcp-2020366
    res_list = list(set(list1) & set(Genlist))

    return res_list
