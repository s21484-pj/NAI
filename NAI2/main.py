"""
Authors: Krzysztof Szymczyk s23210 & Maciej Leciejewski s21484
Our system can determine final grade by attendance, homeworks and final test results.

System requirements:
- Python 3.11
- skfuzzy
- Numpy
"""

import sys
import matplotlib.pyplot as plt
import numpy
import skfuzzy
from skfuzzy import control

NDST = 'Niedostateczny'
DOP = 'Dopuszczajcy'
DST = 'Dostateczny'
DB = 'Dobry'
BDB = 'Bardzo dobry'
ndst_parameter = [0, 0, 40, 50]
dop_parameter = [30, 40, 50, 60]
dst_parameter = [40, 50, 60, 70]
db_parameter = [50, 60, 70, 80]
bdb_parameter = [65, 80, 100, 100]

"""
Parameters:
    - attend (int): Attendance score (0-100)
    - hw (int): Homework score (0-100)
    - test (int): Final test score (0-100)
"""


def compute_fuzzy(attend, hw, test):
    """
    Return the final grade based on attendance, homeworks and final test results.
    Define inputs (attendance, homeworks and final test) and outputs (final grade) for fuzzy logic
    Define membership functions (trapezoidal)
    Define rules for fuzzy logic
    Draws diagram for final grade
    """

    """ 
    Define membership functions for input and output variables
    input rules
    """
    homeworks = skfuzzy.control.Antecedent(numpy.arange(0, 105, 5), 'Homeworks')
    attendance = skfuzzy.control.Antecedent(numpy.arange(0, 105, 5), 'Attendance')
    final_test = skfuzzy.control.Antecedent(numpy.arange(0, 105, 5), 'Final test')

    """output rule"""
    final_grade = skfuzzy.control.Consequent(numpy.arange(0, 105, 5), 'Final grade')

    homeworks[NDST] = skfuzzy.trapmf(homeworks.universe, ndst_parameter)
    homeworks[DOP] = skfuzzy.trapmf(homeworks.universe, dop_parameter)
    homeworks[DST] = skfuzzy.trapmf(homeworks.universe, dst_parameter)
    homeworks[DB] = skfuzzy.trapmf(homeworks.universe, db_parameter)
    homeworks[BDB] = skfuzzy.trapmf(homeworks.universe, bdb_parameter)

    attendance[NDST] = skfuzzy.trapmf(attendance.universe, [0, 0, 45, 55])
    attendance[DOP] = skfuzzy.trapmf(attendance.universe, [35, 45, 55, 65])
    attendance[DST] = skfuzzy.trapmf(attendance.universe, [45, 55, 65, 75])
    attendance[DB] = skfuzzy.trapmf(attendance.universe, [55, 65, 75, 85])
    attendance[BDB] = skfuzzy.trapmf(attendance.universe, [65, 75, 100, 100])

    final_test[NDST] = skfuzzy.trapmf(final_test.universe, ndst_parameter)
    final_test[DOP] = skfuzzy.trapmf(final_test.universe, dop_parameter)
    final_test[DST] = skfuzzy.trapmf(final_test.universe, dst_parameter)
    final_test[DB] = skfuzzy.trapmf(final_test.universe, db_parameter)
    final_test[BDB] = skfuzzy.trapmf(final_test.universe, bdb_parameter)

    final_grade[NDST] = skfuzzy.trapmf(final_grade.universe, ndst_parameter)
    final_grade[DOP] = skfuzzy.trapmf(final_grade.universe, dop_parameter)
    final_grade[DST] = skfuzzy.trapmf(final_grade.universe, dst_parameter)
    final_grade[DB] = skfuzzy.trapmf(final_grade.universe, db_parameter)
    final_grade[BDB] = skfuzzy.trapmf(final_grade.universe, bdb_parameter)

    """Define fuzzy rules"""
    rule1 = skfuzzy.control.Rule(attendance[NDST] & final_test[NDST] & homeworks[NDST], final_grade[NDST])
    rule2 = skfuzzy.control.Rule(attendance[NDST] & final_test[DOP] & homeworks[NDST], final_grade[NDST])
    rule3 = skfuzzy.control.Rule(attendance[NDST] & final_test[DST] & homeworks[NDST], final_grade[DOP])
    rule4 = skfuzzy.control.Rule(attendance[NDST] & final_test[DB] & homeworks[NDST], final_grade[DOP])
    rule5 = skfuzzy.control.Rule(attendance[NDST] & final_test[DST] & homeworks[DB], final_grade[DST])
    rule6 = skfuzzy.control.Rule(attendance[NDST] & final_test[NDST] & homeworks[DOP], final_grade[NDST])
    rule7 = skfuzzy.control.Rule(attendance[NDST] & final_test[DOP] & homeworks[DOP], final_grade[DOP])
    rule8 = skfuzzy.control.Rule(attendance[NDST] & final_test[DST] & homeworks[DOP], final_grade[DOP])
    rule9 = skfuzzy.control.Rule((attendance[NDST] & final_test[DST] & homeworks[DST]), final_grade[DST])
    rule10 = skfuzzy.control.Rule(attendance[NDST] & final_test[BDB] & homeworks[DST], final_grade[DB])
    rule11 = skfuzzy.control.Rule(attendance[DOP] & final_test[DOP] & homeworks[DST], final_grade[DOP])
    rule12 = skfuzzy.control.Rule(attendance[DOP] & final_test[DST] & homeworks[DST], final_grade[DST])
    rule13 = skfuzzy.control.Rule(attendance[DOP] & final_test[DB] & homeworks[DST], final_grade[DST])
    rule14 = skfuzzy.control.Rule(attendance[DOP] & final_test[DB] & homeworks[DB], final_grade[DB])
    rule15 = skfuzzy.control.Rule(attendance[DOP] & final_test[DOP] & homeworks[BDB],
                                  final_grade[DST])
    rule16 = skfuzzy.control.Rule(attendance[DOP] & final_test[DOP] & homeworks[DOP],
                                  final_grade[DOP])
    rule17 = skfuzzy.control.Rule(attendance[DOP] & final_test[NDST] & homeworks[NDST], final_grade[NDST])
    rule18 = skfuzzy.control.Rule(attendance[DOP] & final_test[NDST] & homeworks[DST], final_grade[DOP])
    rule19 = skfuzzy.control.Rule(attendance[DST] & final_test[DOP] & homeworks[DOP], final_grade[DOP])
    rule20 = skfuzzy.control.Rule(attendance[DST] & final_test[BDB] & homeworks[BDB],
                                  final_grade[DB])
    rule21 = skfuzzy.control.Rule(attendance[DST] & final_test[DST] & homeworks[DOP], final_grade[DST])
    rule22 = skfuzzy.control.Rule(attendance[DST] & final_test[NDST] & homeworks[NDST], final_grade[NDST])
    rule23 = skfuzzy.control.Rule(attendance[DB] & final_test[BDB] & homeworks[DB],
                                  final_grade[DB])
    rule24 = skfuzzy.control.Rule(attendance[DB] & final_test[DB] & homeworks[DB], final_grade[DB])
    rule25 = skfuzzy.control.Rule(attendance[DB] & final_test[NDST] & homeworks[NDST], final_grade[NDST])
    rule26 = skfuzzy.control.Rule(attendance[DB] & final_test[DST] & homeworks[DB], final_grade[DB])
    rule27 = skfuzzy.control.Rule(attendance[DB] & final_test[BDB] & homeworks[BDB],
                                  final_grade[BDB])
    rule28 = skfuzzy.control.Rule(attendance[BDB] & final_test[BDB] & homeworks[DB],
                                  final_grade[DB])
    rule29 = skfuzzy.control.Rule(attendance[BDB] & final_test[DOP] & homeworks[DOP],
                                  final_grade[DB])
    rule30 = skfuzzy.control.Rule(attendance[BDB] & final_test[DOP] & homeworks[DB], final_grade[DST])
    rule31 = skfuzzy.control.Rule(attendance[BDB] & final_test[DOP] & homeworks[DST], final_grade[DST])
    rule32 = skfuzzy.control.Rule(attendance[BDB] & final_test[NDST] & homeworks[NDST], final_grade[NDST])
    rule33 = skfuzzy.control.Rule(attendance[BDB] & final_test[DOP] & homeworks[NDST],
                                  final_grade[DOP])
    rule34 = skfuzzy.control.Rule(attendance[BDB] & final_test[NDST] & homeworks[DOP], final_grade[NDST])
    rule35 = skfuzzy.control.Rule(attendance[BDB] & final_test[DST] & homeworks[NDST], final_grade[DST])
    rule36 = skfuzzy.control.Rule(attendance[BDB] & final_test[NDST] & homeworks[DST], final_grade[DOP])
    rule37 = skfuzzy.control.Rule(attendance[BDB] & final_test[DB] & homeworks[NDST], final_grade[DB])
    rule38 = skfuzzy.control.Rule(attendance[BDB] & final_test[NDST] & homeworks[DB], final_grade[DOP])
    rule39 = skfuzzy.control.Rule(attendance[BDB] & final_test[NDST] & homeworks[BDB], final_grade[DST])
    rule40 = skfuzzy.control.Rule(attendance[BDB] & final_test[DOP] & homeworks[BDB],
                                  final_grade[DB])
    rule41 = skfuzzy.control.Rule(attendance[BDB] & final_test[DST] & homeworks[BDB],
                                  final_grade[DB])
    rule42 = skfuzzy.control.Rule(attendance[BDB] & final_test[DB] & homeworks[BDB],
                                  final_grade[DB])
    rule43 = skfuzzy.control.Rule(attendance[BDB] & final_test[BDB] & homeworks[BDB],
                                  final_grade[BDB])

    """Control system setup"""
    rule_list = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14,
                 rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27,
                 rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40,
                 rule41, rule42, rule43]

    performance_ctrl = skfuzzy.control.ControlSystem(rule_list)
    perf_analysis = skfuzzy.control.ControlSystemSimulation(performance_ctrl)

    """Set input values and compute the output"""
    perf_analysis.input['Attendance'] = attend
    perf_analysis.input['Final test'] = test
    perf_analysis.input['Homeworks'] = hw

    perf_analysis.compute()

    """Display the output using Matplotlib"""
    final_grade.view(sim=perf_analysis)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python main.py <attendance> <homework> <final_test>")
    else:
        compute_fuzzy(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))