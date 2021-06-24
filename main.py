import r2d2.R2 as R2D2
import csv
from user import *
import pandas as pd


class Main:
    def __init__(self):
        print("")


if __name__ == '__main__':

    # Fetch data to train on.
    R2 = R2D2.R2()

    # sql = R2.table('exercise_pages').where('id', 24).first()
    sql = R2.table('homeworks_completed',
                   'exercise_id, user_id, teacher_id, completed, result',
                   "TIMEDIFF(completed, started) > '00:15:00'").limit(1000000000)

    user_dict = {}
    # teacher_dict = {}

    records = sql.get()

    for record in records:
        user_list = []
        user_tuple = (record[1], record[0], record[2], record[3], record[4])

        if record[1] in user_dict:
            user_dict[record[1]].append(user_tuple)
        else:
            user_list.append(user_tuple)
            user_dict[record[1]] = user_list

        # teacher_list = []
        # teacher_tuple = (record[0], record[1], record[3], record[4])
        # if record[2] in teacher_dict:
        #     teacher_dict[record[2]].append(teacher_tuple)
        # else:
        #     teacher_list.append(teacher_tuple)
        #     teacher_dict[record[2]] = teacher_list

    for user in user_dict:
        user_dict[user].sort(key=lambda x: (x[2], x[3]))

    # for teacher in teacher_dict:
    #     teacher_dict[teacher].sort(key=lambda x: x[2])

    over_15 = 0
    rest = 0
    user_dict_cleaned = {}

    print(len(user_dict.keys()))
    for key in user_dict:
        exercises = len(user_dict[key])
        if exercises >= 15:
            over_15 += 1
            user_dict_cleaned[key] = exercises
        else:
            rest += 1
    print("over 15: ", over_15)
    print("rest: ", rest)
    print(len(user_dict_cleaned.keys()))

    fieldnames = ['user_id', 'exercise_id', 'teacher_id', 'date_completed', 'result']

    filename = 'user_data.csv'
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)
        for user in user_dict_cleaned:
            for data in user_dict[user]:
                csvwriter.writerow(data)

    # fieldnames = ['exercise_id', 'user_id', 'date_completed', 'result']
    # #
    # filename = 'teacher_data.csv'
    # with open(filename, 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #
    #     csvwriter.writerow(fieldnames)
    #     for teacher in teacher_dict:
    #         for data in teacher_dict[teacher]:
    #             csvwriter.writerow(data)
