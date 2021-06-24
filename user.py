


class User:
    def __init__(self, exercise_id, teacher_id, date_completed, result):
        self.exercise_id = exercise_id
        self.teacher_id = teacher_id
        self.date_completed = date_completed
        self.result = result

    def __str__(self):
        return "'exercise_id': " + str(self.exercise_id) + ", 'teacher_id': " + str(self.teacher_id) + ", 'data_completed:' " + str(self.date_completed) + ", 'sresult:' " + str(self.result)

class Teacher:
    def __init__(self, exercise_id, user_id, date_completed, result):
        self.exercise_id = exercise_id
        self.user_id = user_id
        self.date_completed = date_completed
        self.result = result

#tmo