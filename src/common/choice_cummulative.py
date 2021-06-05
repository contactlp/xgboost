# import random


# def normalize(lst):
#     """Normalize List

#     Args:
#         lst (list): int/float list

#     Returns:
#         list: return noramlzed list

#     print(normalize([1, 2, 3]))
#     [0.16666666666666666, 0.3333333333333333, 0.5]
#     print(normalize([0.16666666666666666, 0.3333333333333333, 0.5]))
#     [0.16666666666666666, 0.3333333333333333, 0.5]
#     """
#     total = float(sum(lst))
#     lst = [i/total for i in lst]

#     return lst


# def cumulative(lst):
#     """convert list in running/cummulative total

#     Args:
#         lst (list[int/float]): list made of int or float

#     Returns:
#         list: return list which has running total.

#     print(cumulative([1, 2, 3]))
#     # [1.0, 3.0, 6.0]
#     print(cumulative([0.16666666666666666, 0.3333333333333333, 0.5]))
#     # [0.16666666666666666, 0.5, 1.0]
#     """

#     output_lst = []
#     running_total = 0.0
#     for i in lst:
#         running_total += i
#         output_lst.append(running_total)

#     return (output_lst)


# def find_index_less_or_equal(float_lst, float_n):
#     """find last index of list where float_n is smaller or equal to

#     Args:
#         float_lst (list[float]): float list
#         float_n (float): find index for this number from float_lst

#     Returns:
#         int: index of float_lst based on float_n

#     print(find_index_less_or_equal(
#         [0.7, 0.8999999999999999, 0.9999999999999999], 0.9))
#     # 1
#     print(find_index_less_or_equal(
#         [0.7, 0.8999999999999999, 0.9999999999999999], 0.8))
#     # 0
#     """
#     index = 0
#     for i in range(len(float_lst)):
#         item = float_lst[i]
#         if float_n >= item:
#             index = i+1
#         else:
#             break
#     return index


# def choice(lst, p, replace=False):
#     """select element from list based on p probability distrubution.

#     Args:
#         lst (list[int/float]): list made of int or float
#         p (int/float): probability distribution.
#         replace (bool, optional): [description]. Defaults to False.

#     Returns:
#         [int/float]: select element [int/float] from lst based on p probability distrubution

#     print(choice([1, 2, 3], [2, 5, 2], replace=False))
#     # 2
#     # 1,2,3 all possible based on probabilities.
#     """

#     if replace != False:
#         return None
#     if len(p) != len(lst):
#         return None

#     normalized_p = normalize(p)
#     normalized_cumulative_p = cumulative(normalized_p)
#     random_float_n = random.random()  # float number between 0.0 to 1.0

#     index = find_index_less_or_equal(normalized_cumulative_p, random_float_n)

#     # print("""
#     # lst                     : %s
#     # p                       : %s
#     # normalized_p            : %s
#     # normalized_cumulative_p : %s
#     # random_float            : %s
#     # index                   : %s
#     # """ % (lst, p, normalized_p, normalized_cumulative_p, random_float_n, index)
#     #       )
#     # print(index)

#     return index


# def choice_n(n, lst, p, replace=False):
#     output = []

#     for i in range(n):
#         index = choice(lst, p, replace=False)
#         selected = lst[index]
#         output.append(selected)
#         del lst[index]
#         del p[index]
#     return output


# def test_choice(lst=[1, 2, 3], p=[7, 2, 1], replace=False):
#     """test choice function

#     Args:
#         lst (list, optional): input list. Defaults to [1, 2, 3].
#         p (list, optional): input list elements probabilities. Defaults to [7, 2, 1].
#         replace (bool, optional): [description]. Defaults to False.

#     Returns:
#         dict: return probabilities of getting each element over 1000 round
#             example: {1: 711,2: 179, 3: 110 }
#     """
#     d = {}

#     for i in range(1000):
#         c = choice(lst, p, replace=False)
#         if c not in d:
#             d[c] = 1
#         else:
#             d[c] += 1
#     return (d)


# def test_choice_n(n=2, lst=[1, 2, 3, 4, 5], p=[10, 5, 3, 2, 1], replace=False):
#     return sorted(choice_n(n, lst, p))


# print(test_choice())
# print(test_choice_n())
