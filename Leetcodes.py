# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:12:48 2018

@author: nkieu
"""
class Solution_TwoSum:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        x_sorted = sorted(nums)
        low, high = 0, len(x_sorted) -1
        while high > low:
            currentSum = x_sorted[low] + x_sorted[high]
            print(currentSum)
            if  currentSum == target:
                break
            elif currentSum > target:
                high += -1
            else:
                low += 1
        
        low_res = nums.index(x_sorted[low])
        high_res = nums.index(x_sorted[high])
        
        if x_sorted[low] == x_sorted[high]:
            high_res = len(nums) - nums[::-1].index(x_sorted[high]) - 1
            
            
        return(sorted([low_res, high_res]))
    
    def test(self):
        x = [3,3]
        print(self.twoSum(x, 6))

class Solution_TwoArrays:
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None
            
    def addTwoNumbers(self, l1, l2):        
        result = []
        
        toAdd = 0
        tmp1, tmp2 = l1, l2
        
        while True:
            result.append((tmp1.val + tmp2.val + toAdd) % 10)
            toAdd = 1 if tmp1.val + tmp2.val + toAdd >= 10 else 0 
            
            if not (toAdd != 0 or tmp1.next != None or tmp2.next != None):
                break
            
            tmp1 = tmp1.next if tmp1.next != None else ListNode(0)
            tmp2 = tmp2.next if tmp2.next != None else ListNode(0)
                        
        return(result)
            
    def test1(self):
#        Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
#        Output: 7 -> 0 -> 8
#        Explanation: 342 + 465 = 807.
        l1 = ListNode(2)
        l1.next = ListNode(4)
        l1.next.next = ListNode(0)
        l1.next.next.next = ListNode(4)
        
        l2 = ListNode(5)
        l2.next = ListNode(4)
        l2.next.next = ListNode(0)
        l2.next.next.next = ListNode(6)
        
        print(self.addTwoNumbers(l1, l2))
    
    def test2(self):
#        Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
#        Output: 7 -> 0 -> 8
#        Explanation: 342 + 465 = 807.
        l1 = ListNode(2)
        l1.next = ListNode(4)
        l1.next.next = ListNode(3)
        
        l2 = ListNode(5)
        l2.next = ListNode(6)
        l2.next.next = ListNode(4)
        
        print(self.addTwoNumbers(l1, l2))
    
    def test3(self):
#        Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
#        Output: 7 -> 0 -> 8
#        Explanation: 342 + 465 = 807.
        l1 = ListNode(0)
        l2 = ListNode(0)
        print(self.addTwoNumbers(l1, l2))

class Solution_CountChange:
    '''
    Number of ways to make changes given a list of coins of decreasing denomination
    '''
    def Solve(self, money, denominations):
        if money < 0:
            return 0
        elif money == 0:
            return 1
        elif len(denominations) == 0:
            return 0
        else:
            return (self.Solve(money - denominations[0], denominations) # number of ways to make change using 1 coin or largest denomination
                    + self.Solve(money, denominations[1:])) # number of ways to make change without using the largest nomination
    
    def test(self):
        assert(self.Solve(4, [3,2,1]) == 4)
        assert(self.Solve(10, [2,3,5,6]) == 5)

class Solution_CountChangePrint:
    '''
    Print out the different ways to make change given a list of coins of decreasing denomination
    '''
    def Solve(self, money, denominations):
        if money < 0:
            return []
        elif money == 0:
            if len(denominations) > 0:
                return [[]]
            else:
                return []
        elif len(denominations) == 0:
            return []
        else:
            return ([[denominations[0]] + way for way in self.Solve(money - denominations[0], denominations)] # number of ways to make change using 1 coin or largest denomination
                    + self.Solve(money, denominations[1:])) # number of ways to make change without using the largest nomination
    
    def test(self):
        print(self.Solve(4, [3,2,1]))
        print(self.Solve(10, [2,3,5,6]))
        
class Solution_PrintPermutation:
    def Solve(self, nums):
        if len(nums) == 0:
            return [[]]
        else:
            result = []
            for i in range(len(nums)):
                sub_list = nums[:i] + nums[(i+1):]
                
                for way in self.Solve(sub_list):
                    result.append([nums[i]] + way)
            return(result)
            
    def test(self):
        result = self.Solve(list(range(5)))
        assert(len(result) == 120)

class CombinationSum:
    '''
    Just like the counting coin problem
    '''
    def Solve(self, nums, target):
        print(nums, target)
        if target < 0:
            return []
        elif target == 0:
            return [[]]
        elif len(nums) == 0:
            return []
        elif target > sum(nums):
            return []
        elif target == sum(nums):
            return [nums]
        else:
            tmp_res = []
            for tmp in self.Solve(nums[1:], target - nums[0]):
                tmp_res.append([nums[0]] + tmp)
            return tmp_res + self.Solve(nums[1:] , target)
    
    def test(self):
        # Current issues: return duplicate solutions if there are duplicate
        return(self.Solve(sorted([10, 1, 2, 7, 6, 1, 5])[::-1],
                   8))
        
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:49:01 2018

@author: nkieu
"""

class MaximumIndex_2:
    '''
    Given an array A of integers, find the maximum of j - i subjected to the constraint of A[i] <= A[j].

    Example :
    
    A : [3 5 4 2]
    
    Output : 2 
    for the pair (3, 4)
    
    '''
    def Solve(self, nums):
        # LMin[i]: index of the lowest value to the left of i
        # RMax[j]: index of the highest value to the right of j
        LMin, RMax = [0] * len(nums), [len(nums) - 1] * len(nums)
        for i in range(1, len(nums)):
            if nums[i] < nums[LMin[i-1]]:
                LMin[i] = i
            else:
                LMin[i] = LMin[i-1]
        
        # LMin and RMax are both increasing series
        # at each point in LMin and Rmax: we know the lowest index to the left, and the highest point to the right
        # if the values at these points satisfy A[i] < A[j]
        for i in range(len(nums)-2,-1, -1):
            if nums[i] > nums[RMax[i+1]]:
                RMax[i] = i
            else:
                RMax[i] = RMax[i+1]
        
        maxVal, indexValue = 0, 0
        i = 0
        while i < len(nums):
            if nums[LMin[i]] < nums[RMax[i]]:
                if maxVal < RMax[i] - LMin[i]:
                    maxVal = RMax[i] - LMin[i]
                    indexValue = i
            i += 1
        return LMin[indexValue], RMax[indexValue]
    
    def test(self):
        nums = [6,5,4,3,1,5]
        # nums = [6,5,3,4,3,2,1,4]
        result = self.Solve(nums)
        print(nums[result[0]], nums[result[1]], result[1] - result[0])


import numpy as np

class MaximumIndex:
    '''
    Given an array A of integers, find the maximum of j - i subjected to the constraint of A[i] <= A[j].

    Example :
    
    A : [3 5 4 2]
    
    Output : 2 
    for the pair (3, 4)
    
    '''
    def Solve(self, nums):
        argindex = list(np.argsort(nums))
        
        tmp = [0] * (len(argindex)- 1 )
        for i in range(len(tmp)):
            tmp[i] = argindex[i+1] - argindex[i]
        
        # Solve for largest sub-array
        first, last = LargestSubarray().Solve(tmp)
        return nums[argindex[first+1]], nums[argindex[last + 1]]
        # Find the largest subarray of tmp
        
    
    def test(self):
        nums = [6,5,4,3,1,5]
        # nums = [6,5,3,4,3,2,1,4]
        result = self.Solve(nums)
        print(nums[result[0]], nums[result[1]], result[1] - result[0])
        
class LargestSubarray:
    def Solve(self, nums):
        # Kadane's algorithm
        # source: wikipedia
        max_ending_here = max_so_far = nums[0]
        startIndex, endIndex = 0, 0
        tmp_start, tmp_end = 0, 0
        
        for i in range(1, len(nums)):
            # basically compare:
            #   current subarray
            #   current subarray + next element
            #   next element only
            if nums[i] >= max_ending_here + nums[i]:
                tmp_start = i
                tmp_end = i
                max_ending_here = nums[i]
            else:
                tmp_start = startIndex
                tmp_end = i
                max_ending_here = max_ending_here  + nums[i]
            
            if max_so_far <= max_ending_here:
                startIndex = tmp_start
                endIndex = tmp_end
                max_so_far = max_ending_here
                
        return startIndex, endIndex
    
    def test(self):
        nums = [1, -5, 1, 1, -2]
        print(self.Solve(nums))

class Solution_lengthOfLongestSubstring:
    '''
    Given a string, find the length of the longest substring without repeating characters.
    Examples:
        Given "abcabcbb", the answer is "abc", which the length is 3.
        Given "bbbbb", the answer is "b", with the length of 1.
        Given "pwwkew", the answer is "wke", with the length of 3. 
        Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
        
    Note: this solution is still not good enough. I would like to avoid using function .index
    '''
    def Solve(self, st):
        s = list(st)

        currentLen, maxLen = 1, 1
        currentInd = 0
        currentSet = s[:1]
        maxInd = 0
        
        for i in range(1, len(s)):
            if s[i] not in currentSet:
                currentSet.append(s[i])
                currentLen += 1
            else:
                currentInd = currentInd + currentSet.index(s[i]) + 1
                currentSet = s[currentInd:(i+1)]
                currentLen = len(currentSet)
                
            if currentLen >= maxLen:
                maxLen = currentLen
                maxInd = currentInd
                
        return maxLen, st[maxInd:(maxInd+maxLen)]
#        return maxIndex, maxLen
    
    def test(self):
        print(self.Solve("abcabcb"))
        print(self.Solve("abccabcde"))
        print(self.Solve("bbbbb"))
        print(self.Solve("pwkew"))


class NextPermutation:
    '''
    Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.
    If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).
    The replacement must be in-place and use only constant extra memory.
    Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.
    
    1,2,3 → 1,3,2
    3,2,1 → 1,2,3
    1,1,5 → 1,5,1
    
    Solution: https://www.nayuki.io/page/next-lexicographical-permutation-algorithm
    '''
    def Solve(self, current):
        # find the first pair that has the property of left < right
        # switch between left and right
        # sort everything from left+1 to the end in ascending order
        i = len(current)-1
        j = len(current)-1
        while i >= 0 and current[i] <= current[i-1]:
            i += -1 

        i += -1
        
        while j >= (i+1) and current[j] <= current[i]:
            j = j -1

        # Swap
        current[i], current[j] = current[j], current[i]
        
        #Reverse
        current[(i+1):] = current[(i+1):][::-1]
        return(current)
        

    def test(self):
        print(self.Solve([0,1,2,5,3,3,0]))
        # expect 0 1 3 0 2 3 5  

class TowerOfHanoi:
    def __init__(self, n):
        source = list(range(n))
        source.reverse()
    
        self.pegs = [source, [], []]
        
    def Solve(self, NumToMove, source, other, dest):
        
        if NumToMove == 1:
            tmp = self.pegs[source].pop()
            self.pegs[dest].append(tmp)
            print(self.pegs)
        else:
            self.Solve(NumToMove -1, source, dest, other)
            
            tmp = self.pegs[source].pop()
            self.pegs[dest].append(tmp)
            print(self.pegs)
            self.Solve(NumToMove -1, other, source, dest)
    
    def SolveAll(self):
        self.Solve(n, 0, 1, 2)
