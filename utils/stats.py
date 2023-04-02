"""
Uses methods from DeepCrime: https://github.com/dlfaults/deepcrime/blob/f1f4c8b2d26a1dfb75674cbac0ebf11fa62ec98a/stats.py
"""

import numpy as np
import statsmodels.api as sm


def cohen_d(orig_accuracy_list, accuracy_list):
    """
    Calculates cohen's kappa value.
    """
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        (
            (nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2
            + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2
        )
        / dof
    )
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result


def is_significant(orig_accuracy_list, accuracy_list, threshold=0.05):

    p_value = p_value_glm(orig_accuracy_list, accuracy_list)
    effect_size = cohen_d(orig_accuracy_list, accuracy_list)
    is_significant = (p_value < threshold) and effect_size >= 0.5

    return is_significant, p_value, effect_size


def p_value_glm(orig_accuracy_list, accuracy_list):
    list_length = len(orig_accuracy_list)

    zeros_list = np.zeros(list_length)
    ones_list = np.ones(list_length)
    mod_lists = np.concatenate((zeros_list, ones_list))
    acc_lists = np.concatenate((orig_accuracy_list, accuracy_list))

    response = acc_lists
    predictors = sm.add_constant(mod_lists)

    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value_g = float(pv)

    return p_value_g
